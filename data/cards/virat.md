# Data Card — VIRAT Ground (alternative source)

> A secondary source considered for SENTRY. This card is shorter than the
> [UCF-Crime card](ucf_crime.md) (the primary source); it documents what VIRAT
> offers and why it is a complement, not a replacement.

## At a glance

| | |
|---|---|
| **Name** | VIRAT Video Dataset (Ground component) |
| **Sponsor** | DARPA / IARPA; released via the VIRAT consortium |
| **Published** | Oh et al., "A Large-scale Benchmark Dataset for Event Recognition in Surveillance Video", CVPR 2011 |
| **Modality** | Stationary outdoor surveillance (parking lots, building approaches) |
| **Size** | ~**8.5 hours** ground video, HD, 11 scenes |
| **Labels** | **Event annotations** (event type + bounding box + start/end frame) and object tracks — e.g. *person loading/unloading an object from a vehicle, person opening a trunk, person entering/exiting a facility, vehicle turning* |
| **Project page** | https://viratdata.org/ |

## What's distinctive

- **Government-released, clean annotation contract.** Bounding boxes + event
  types + frame spans + object tracks — the richest *structured* labels of the
  three, ideal for spatially-grounded, typed reports.
- **Realistic "quiet" surveillance.** Mostly mundane outdoor activity with
  occasional events of interest — closer to the real SOC base rate (lots of
  normal, rare signal) than UCF-Crime's anomaly-dense montage.
- **Consistent capture.** Fixed elevated cameras, stable framing → good for
  studying camera-specific behavior and per-location splits.

## Why it's a complement, not the primary source for SENTRY

- **Activities, not crimes.** VIRAT events are everyday actions (loading a car,
  walking, gesturing), not the 13 public-safety incident types SENTRY targets,
  so it yields fewer "incident report" exemplars.
- **Lower visual drama / class overlap** with SENTRY's intended use, though its
  *abandoned-object* and *vehicle* events map cleanly onto SENTRY's taxonomy.
- **Best fit:** a **structured-grounding / per-camera-split** set, and a source
  of realistic *normal* footage to balance UCF-Crime's anomaly skew.

## Known biases & limitations

- North-America-centric scenes (US parking lots / facilities); limited diversity
  of environment and signage.
- Sparse events across long footage → heavy class imbalance toward "nothing
  happening" (realistic, but needs sampling care).
- Elevated, distant cameras → small actors, limited appearance detail.

## License & access

Released for research use via viratdata.org (terms apply). **Not redistributed
here.** No adapter is shipped for it yet — see *roadmap* in the
[cards index](README.md).

## Citation

```bibtex
@inproceedings{oh2011virat,
  title     = {A Large-scale Benchmark Dataset for Event Recognition in Surveillance Video},
  author    = {Oh, Sangmin and others},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2011}
}
```
