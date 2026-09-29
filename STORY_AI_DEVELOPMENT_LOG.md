# STORY-AI Development Log

This is the canonical research log. Each entry follows the same evidence chain:

```text
problem → hypothesis → experiment → observation → interpretation → decision → next gate
```

Historical, detailed versions remain in [docs/archive/](docs/archive/).

## 2026-09-28 — Perception and OCR baseline

**Problem.** Manga text cannot be treated as a flat OCR stream: CTD regions may be duplicated, rotated, fragmented, or unrelated to story text.

**Experiment.** Run Comic Text Detector for localization, generate deterministic polygon crops, and compare PaddleOCR recognition with and without rectification, upscaling, padding, and rotation.

**Evidence.** CTD localized useful regions but its language metadata was not reliable as an English filter. Geometry materially improved horizontal recognition. Difficult cases remained in vertical/tiny text and ordering/grouping.

**Decision.** CTD is a localizer; OCR recognition and semantic story filtering are separate stages. Preserve multiple OCR hypotheses in the Candidate Bank.

**Next gate.** Evaluate the complete sequence pipeline against labelled development data rather than optimizing isolated OCR examples.

## 2026-09-28 — Learned preprocessing investigation

**Problem.** A preprocessing policy might reduce expensive OCR retries.

**Experiment.** Build an oracle from development labels by scoring `NONE`, `UPSCALE`, `CONTRAST`, `THRESHOLD`, and rotation variants with character error rate.

**Evidence.** The corrected oracle made `NONE` a real no-op, removed duplicate CTD blocks, and used one-to-one ground-truth matching. On the inspected sample, high error often came from line order or segmentation rather than image quality; transformations frequently made already-good text worse.

**Decision.** Do not make a zero-shot learned preprocessing controller part of production. The historical Laya investigation is retained in the archive, but Laya is absent from executable architecture.

**Next gate.** Treat reading order and balloon grouping as explicit structured problems and measure them on development sequences.

## 2026-09-28 — Semantic structure and evidence fusion

**Problem.** OCR candidates alone cannot reliably decide whether a region is story text, SFX, narration, or a balloon fragment, nor can they assign a speaker.

**Experiment.** Introduce panel/balloon grouping, balloon-level crops, Candidate Bank evidence, Qwen3-VL adjudication, character perception, speaker grounding, reading order, identity resolution, and a sequence validator.

**Evidence.** The architecture supports independent provenance, deterministic crop identities, explicit ambiguity, sequence-local character labels, and schema-level diagnostics.

**Decision.** Qwen3-VL is the image-primary balloon adjudicator. Geometry and visual context ground speakers; identity is a sequence-local graph; LangGraph remains orchestration only.

**Next gate.** Run labelled development evaluation and report text, order, speaker, identity, and contract metrics separately.

## 2026-09-29 — Integration baseline recorded

**Evidence.** The established regression baseline is 212/212 tests passing. The recorded three-page integration baseline contains 34 CTD regions, 29 balloons, 119 candidates, 31 characters, 15 resolved speakers, 14 narration items, 12 identity clusters, 318 identity pairs, 29 reading-order items, zero resolver diagnostics, and zero validation errors.

**Interpretation.** This establishes execution and contract validity, not competition accuracy. Qwen adjudication is the largest measured runtime stage; Nemotron depends on the WSL worker.

**Decision.** Keep the architecture locked while moving to development-set scoring and per-model runtime telemetry.

## Research status

| Area | Status | Evidence or next action |
|---|---|---|
| Data and schemas | Complete | Regression-covered contracts |
| CTD localization | Integrated | Local ONNX detector |
| Layout and balloon grouping | Integrated | YOLO plus geometry fallback |
| Candidate Bank | Integrated | Independent provenance preserved |
| Qwen adjudication | Integrated | Balloon-level image-primary decision |
| Character and speaker reasoning | Integrated | Regression-covered |
| Identity and sequence resolution | Integrated | Zero diagnostics in baseline |
| Development-set evaluation | Next gate | Run official scorer and error taxonomy |
| Runtime instrumentation | Open | Add per-model, per-stage timings |
