# Data Card — ShanghaiTech Campus (alternative source)

> A secondary source considered for SENTRY. This card is shorter than the
> [UCF-Crime card](ucf_crime.md) (the primary source); it documents what
> ShanghaiTech offers and why it is a complement, not a replacement.

## At a glance

| | |
|---|---|
| **Name** | ShanghaiTech Campus dataset |
| **Authors** | Weixin Luo, Wen Liu, Shenghua Gao — ShanghaiTech University |
| **Published** | "A Revisit of Sparse Coding Based Anomaly Detection in Stacked RNN Framework", ICCV 2017 |
| **Modality** | Fixed-camera campus surveillance video |
| **Size** | **13 scenes / camera viewpoints**, 130+ abnormal events, ~330 training + 100+ testing clips |
| **Labels** | **Pixel- and frame-level** anomaly masks (where + when), no class names |
| **Project page** | https://svip-lab.github.io/dataset/campus_dataset.html |

## What's distinctive

- **Multi-scene, fixed cameras.** 13 real campus cameras with different angles
  and lighting — closer to a *single-deployment* surveillance setting than
  UCF-Crime's web montage.
- **Precise spatial localization.** Anomalies are masked at the **pixel** level,
  which is excellent for grounding/attention evaluation (SENTRY's Grad-CAM
  overlay could be quantitatively scored against these masks).
- **Scene-defined anomalies.** "Anomaly" = anything not seen in that scene's
  normal training footage (cyclists/cars on a pedestrian walkway, fighting,
  loitering, jumping) — so it captures *contextual* anomalies UCF-Crime lacks.

## Why it's a complement, not the primary source for SENTRY

- **No incident categories or text.** Anomalies are unlabeled-by-type binary
  masks, so it gives weaker supervision for *report* content than UCF-Crime's 13
  named classes. SENTRY is about typed incident reports, so UCF-Crime leads.
- **Narrower variety.** One campus, staged-ish anomalies → far less incident and
  environment diversity than UCF-Crime's real crime footage.
- **Best fit:** a **grounding-evaluation** set — score whether the model's
  attention lands on the pixels ShanghaiTech marks as anomalous.

## Known biases & limitations

- Single institution, single climate/architecture → strong domain skew.
- Many anomalies are **enacted** for the dataset (people told to run/fight), not
  organic, so motion cues may be unnaturally clean.
- Daytime-dominant; limited night/low-light coverage.

## License & access

Academic research use; obtain from the official project page. **Not redistributed
here.** No adapter is shipped for it yet — see *roadmap* in the
[cards index](README.md).

## Citation

```bibtex
@inproceedings{luo2017revisit,
  title     = {A Revisit of Sparse Coding Based Anomaly Detection in Stacked RNN Framework},
  author    = {Luo, Weixin and Liu, Wen and Gao, Shenghua},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2017}
}
```
