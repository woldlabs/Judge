# Vision — Judge

## Product purpose
Judge (Joint Unconventional Data & Geophysical Examination) is a multimodal anomaly detection framework for identifying and characterizing statistically unusual or physically unconventional events in video, audio, and sensor datasets. It serves researchers, field investigators, and data scientists working observational recordings where transient, non-stationary, or cross-modal signatures may indicate rare phenomena.

Wold Labs north star: secure the future of humanity with innovative, practical, accessible solutions rooted in first principles. Judge contributes by turning raw multi-instrument recordings into explainable, reproducible anomaly candidates — extraordinary claims need extraordinary data hygiene.

## Non-goals
- Not a deep-learning black-box classifier as the primary path; signal-processing-first, interpretable features win by default.
- Not a geospatial aggregation or web-intel product (that is Rift).
- Not a production SaaS / hosted pipeline in this repo; local/desktop + headless CLI analysis.
- Does not assert that detections are confirmed phenomena — all events are statistical candidates only.
- Does not invent cybersecurity product claims; scope is anomaly detection on observational media/sensors.

## Architecture boundaries (change only with review)
- `BaseDetector` contract and detector plugin layout under `judge/core/detectors/`.
- Cross-modal fusion semantics: keep supporting detections; score-boost + tag coincidences (do not silently drop).
- Event model / report JSON schema used by exporters and by Rift’s Judge bridge.
- Conservative defaults: high specificity; user-controlled sensitivity, min duration, coincidence window.
- CPU-only, reproducible seeded paths where applicable; do not hard-require GPU/CUDA.

## Success metrics
- pytest suite green on CI (Python 3.10–3.12) for every PR to main.
- Headless `analyze` / `demo` / `version` CLI remain usable without GUI.
- Reports (PDF/MD/JSON) remain round-trippable and include reproducibility footer.
- Cross-modal coincidences remain countable and tagged without collapsing supporting events.
- New detectors implement `BaseDetector` without forking core session/fusion.

## Relation to Rift
Rift finds *where* to look (sites, portals, fracture clusters on a map). Judge analyzes *what was recorded* at those places (video/audio/sensor). Rift exports Judge-ready manifests and can re-import Judge report JSON as map pins. Changes to Judge’s report/event schema need a paired check of `rift/integrations/judge.py`.
