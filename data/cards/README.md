# Dataset cards

Documentation of every dataset SENTRY uses or evaluated, so the provenance and
bias of the training signal are inspectable without opening a video. Each card
covers **what the dataset contains, how it was collected, and its known
limitations** — the part most portfolio projects skip.

| Card | Role | Why |
|---|---|---|
| [**ucf_crime.md**](ucf_crime.md) | **Primary source** ✅ | 1,900 real CCTV videos, 13 named incident classes + temporal windows → real variety with labels that map onto SENTRY's incident taxonomy. **Integrated** (`data.datasets.ucf_crime`). |
| [shanghaitech.md](shanghaitech.md) | Complement (grounding eval) | Pixel-level anomaly masks across 13 fixed campus cameras — ideal for *scoring* SENTRY's Grad-CAM grounding, but no incident types/text. |
| [virat.md](virat.md) | Complement (structured normal) | Government-released event boxes + tracks; realistic "quiet" outdoor surveillance — good for per-camera splits and balancing UCF-Crime's anomaly skew. |

## Why UCF-Crime is the primary source

SENTRY generates **typed incident reports with a reliability flag.** That goal
drove the choice:

| Need | UCF-Crime | ShanghaiTech | VIRAT |
|---|---|---|---|
| Named **incident categories** (for typed reports + event-F1) | ✅ 13 classes | ❌ binary anomaly | ⚠️ activities, not crimes |
| Real-world **variety** of incidents/scenes | ✅ web-wide | ❌ one campus | ⚠️ a few sites |
| **Temporal** localization (when) | ✅ test set | ✅ frame masks | ✅ event spans |
| **Spatial** localization (where) | ❌ | ✅ pixel masks | ✅ boxes |
| Natural-language **descriptions** | ❌ (templated) | ❌ | ❌ |
| Scale | ✅ 1,900 / 128 h | ⚠️ ~430 clips | ⚠️ ~8.5 h |

UCF-Crime wins on the axes that matter most for *typed* report generation
(categories + variety + scale); the other two are noted as complements for
**grounding evaluation** (ShanghaiTech) and **realistic normal / per-camera
splits** (VIRAT). None of the three ship natural-language reports, which is why
SENTRY templates reports from labels and treats them as weak supervision — see
the *Reports & weak supervision* section of the UCF-Crime card.

## Roadmap

- [x] UCF-Crime adapter + manifest (`src/data/datasets/ucf_crime.py`)
- [ ] ShanghaiTech adapter scoped as a **grounding-evaluation** set (score
      Grad-CAM against pixel masks)
- [ ] VIRAT adapter for **per-camera/location splits** and normal-footage balancing
- [ ] Pair UCF-Crime with a captioned surveillance source for true free-text
      report supervision

## Conventions for a SENTRY data card

Each card answers, in this order: **At a glance** (provenance table) · **What's
in it** · **How it was collected** · **How SENTRY processes it** · **Known biases
& limitations** · **License & access** · **Ethics & intended use** · **Citation**.
Keep biases concrete (class imbalance numbers, sourcing method, geographic skew)
— that section is the point.
