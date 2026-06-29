"""Evaluation metrics for generated incident reports.

`compute_text_metrics` / `benchmark_metrics` are domain-agnostic NLG metrics.
`event_overlap` is an incident-aware metric: does the generated report mention the
same key events as the reference? More meaningful than BLEU, which rewards fluent
boilerplate regardless of facts.
"""

from __future__ import annotations


def compute_text_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute BLEU, ROUGE-L and METEOR for report generation."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")

    scores: dict[str, float] = {}

    # BLEU (corpus-level) via sacrebleu
    import sacrebleu

    scores["bleu"] = sacrebleu.corpus_bleu(predictions, [references]).score

    # ROUGE-L (mean F-measure) via rouge_score
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    scores["rougeL"] = 100.0 * sum(rouge_l) / len(rouge_l)

    # METEOR via nltk
    import nltk
    from nltk.translate.meteor_score import meteor_score

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    meteor = [meteor_score([r.split()], p.split()) for p, r in zip(predictions, references)]
    scores["meteor"] = 100.0 * sum(meteor) / len(meteor)

    return scores


# --- incident-event overlap -------------------------------------------------
# "Does the generated report mention the same key events as the reference?"
# A keyword lexicon of common surveillance incident categories, with light
# negation handling. STARTING POINT — refine for your incident taxonomy.

EVENT_CATEGORIES: dict[str, list[str]] = {
    "intrusion": ["intrusion", "intruder", "trespass", "unauthorized entry", "break-in", "breaking in"],
    "loitering": ["loiter", "loitering", "lingering"],
    "theft": ["theft", "stealing", "shoplifting", "robbery", "burglary"],
    "weapon": ["weapon", "gun", "firearm", "knife", "armed", "pistol", "rifle"],
    "violence": ["fight", "assault", "altercation", "attack", "violence", "punch"],
    "vandalism": ["vandalism", "graffiti", "damage", "destruction"],
    "abandoned object": ["abandoned", "unattended", "left behind", "suspicious package", "suspicious bag"],
    "crowd": ["crowd", "gathering", "group of people"],
    "fire/smoke": ["fire", "smoke", "flames"],
    "vehicle": ["vehicle", "car", "truck", "motorcycle", "van"],
    "fall": ["fall", "fell", "collapsed", "lying on the ground"],
    "tailgating": ["tailgating", "tailgate", "following closely"],
}
_NEG_CUES = ("no ", "without", "negative for", "free of", "no sign", "no evidence",
             "absence of", "not ", "nothing", "all clear")


def _mentioned_events(text: str) -> set[str]:
    """Events whose terms appear at all (regardless of polarity)."""
    t = text.lower()
    return {e for e, kws in EVENT_CATEGORIES.items() if any(k in t for k in kws)}


def _asserted_events(text: str) -> set[str]:
    """Events asserted as present (term appears and is not locally negated)."""
    t = text.lower()
    out: set[str] = set()
    for event, kws in EVENT_CATEGORIES.items():
        for k in kws:
            idx = t.find(k)
            if idx == -1:
                continue
            window = t[max(0, idx - 30):idx]
            if not any(cue in window for cue in _NEG_CUES):
                out.add(event)
            break
    return out


def _micro_prf(pred_sets, ref_sets) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for p, r in zip(pred_sets, ref_sets):
        tp += len(p & r)
        fp += len(p - r)
        fn += len(r - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def event_overlap(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Micro-averaged event overlap: mention-based (headline) + asserted-only."""
    mp, mr, mf = _micro_prf(
        [_mentioned_events(p) for p in predictions],
        [_mentioned_events(r) for r in references],
    )
    _, _, af = _micro_prf(
        [_asserted_events(p) for p in predictions],
        [_asserted_events(r) for r in references],
    )
    return {
        "event_f1": 100.0 * mf,           # headline: same events mentioned?
        "event_precision": 100.0 * mp,
        "event_recall": 100.0 * mr,
        "event_asserted_f1": 100.0 * af,  # stricter: present-events agreement
    }


def score(predictions: list[str], references: list[str]) -> dict[str, float]:
    """All metrics: text overlap (BLEU/ROUGE-L/METEOR) + event overlap."""
    return {**compute_text_metrics(predictions, references),
            **event_overlap(predictions, references)}


# --- paper-comparable NLG metrics (for benchmarking against published numbers) ---
# Report/caption-generation papers report cumulative BLEU-1..4, METEOR and ROUGE-L
# on lowercased, word-tokenized text (COCO/nltk-style), on a 0-1 scale.
# compute_text_metrics above uses sacrebleu (different tokenization/smoothing), so
# it is NOT comparable to those; use this for head-to-head benchmarks.

def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"\w+", text.lower())


def benchmark_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Cumulative BLEU-1..4, METEOR, ROUGE-L on a 0-1 scale (paper-comparable)."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    hyps = [_tokenize(p) for p in predictions]
    refs = [[_tokenize(r)] for r in references]
    smooth = SmoothingFunction().method1
    out: dict[str, float] = {}
    for n in (1, 2, 3, 4):
        weights = tuple([1.0 / n] * n)
        out[f"bleu{n}"] = corpus_bleu(refs, hyps, weights=weights, smoothing_function=smooth)

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    out["rougeL"] = sum(rl) / len(rl)

    import nltk
    from nltk.translate.meteor_score import meteor_score
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    mt = [meteor_score([_tokenize(r)], _tokenize(p)) for p, r in zip(predictions, references)]
    out["meteor"] = sum(mt) / len(mt)
    return out


if __name__ == "__main__":
    demo_preds = ["an intruder enters through the fence; no weapon visible"]
    demo_refs = ["intruder climbs the fence at the rear; unarmed"]
    print(score(demo_preds, demo_refs))
