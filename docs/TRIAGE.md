# Triage rubric — Judge

Use this checklist when reviewing PRs or issues against [VISION.md](../VISION.md). Prefer purpose fit, non-goals, architecture boundaries, and tests over nits.

## Against VISION.md

- [ ] **Purpose fit** — Does the change serve multimodal anomaly detection on video/audio/sensor observational data (explainable, reproducible candidates)?
- [ ] **Non-goals** — Does it avoid: DL-black-box-as-primary path, geospatial/web-intel (Rift’s job), hosted SaaS scope, treating detections as confirmed phenomena, invented cybersecurity claims?
- [ ] **Architecture boundaries** — Any change to `BaseDetector` / detector layout, cross-modal fusion (keep supporting detections; boost+tag coincidences), event/report JSON schema, conservative defaults, or CPU-only assumptions? If yes → needs explicit review, not drive-by edits.
- [ ] **Success metrics** — Still green: CI pytest (3.10–3.12), headless CLI (`analyze`/`demo`/`version`), report round-trip + reproducibility footer, coincidence tagging without collapsing events, new detectors via `BaseDetector`.
- [ ] **Cross-repo (Judge ↔ Rift)** — Does this touch report/event schema or export shapes consumed by `rift/integrations/judge.py`? If yes, flag Rift follow-up risk before merge.

## PR quality (after vision fit)

- [ ] Clear **testable goal** in title/body (what becomes true when this lands).
- [ ] Risk called out (data loss, secrets, blast radius) or N/A.
- [ ] Tests added/updated when behavior changes.
- [ ] Docs-only changes stay docs-only.

## Labels (suggested)

`needs-vision-review` · `docs` · `schema-risk` (if Judge↔Rift bridge) · `architecture`
