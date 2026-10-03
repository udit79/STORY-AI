# STORY-AI Decisions Register

This is the canonical stable decision register. Historical decisions and superseded alternatives are preserved in [docs/archive/STORY_AI_DECISIONS_v10.md](docs/archive/STORY_AI_DECISIONS_v10.md) and earlier versions.

## D001 — Sequence is the unit of data and identity

**Status: LOCKED.** The pipeline consumes three consecutive pages as one sequence. Train/validation splits preserve sequence boundaries. Character labels are anonymous and sequence-local.

## D002 — Separate localization, recognition, semantics, and relations

**Status: LOCKED.** CTD answers where text-like regions are. OCR models produce evidence about what they may say. Qwen adjudicates balloon meaning and story-text inclusion. Geometry and visual context handle speaker grounding and ordering.

## D003 — CTD language metadata is not a production filter

**Status: LOCKED.** Development evidence showed unreliable language labels. Story-text inclusion requires balloon context, OCR evidence, image evidence, and explicit validation.

## D004 — Candidate Bank preserves hypotheses and provenance

**Status: LOCKED.** Evidence from PaddleOCR, Nemotron, Qwen proposals, preprocessing variants, and fused candidates remains independently traceable. No single confidence score is treated as truth.

## D005 — Qwen3-VL is the balloon-level adjudicator

**Status: LOCKED.** The decision unit is the semantic balloon, not an individual CTD fragment. Qwen receives the balloon image and candidate evidence and may correct jointly-wrong candidates.

## D006 — Reading order is hierarchical

**Status: LOCKED.** The intended hierarchy is page → panel → balloon → text. A global x/y sort is insufficient. Reading-order decisions retain geometry and panel evidence.

## D007 — Speaker grounding is a multimodal relation problem

**Status: LOCKED.** Candidate speakers are generated from relevant characters and resolved with balloon-tail geometry, panel membership, spatial context, visibility, and visual evidence. `UNKNOWN` is valid internal and serialized output when evidence is insufficient.

## D008 — Identity is a sequence-local graph

**Status: LOCKED.** Cross-page character similarity creates an identity graph and anonymous clusters. The system must preserve same-character consistency without collapsing distinct same-page characters.

## D009 — Sequence resolver and validator are explicit gates

**Status: LOCKED.** Final output must have valid source references, acyclic order, valid speaker references, consistent identities, no excluded text, and no invented source regions. JSONL serialization is validated separately from model quality.

## D010 — Runtime orchestration is procedural

**Status: LOCKED.** The supported runner calls the Python pipeline for each three-page sequence. The current runtime does not use LangGraph; component boundaries remain in `app/`, while `tools/pipeline_integration_smoke.py` owns execution order and diagnostics.

## D011 — Laya is historical, not production architecture

**Status: LOCKED.** The archived Laya experiments are retained as research evidence. Laya is absent from production code and the current pipeline uses Candidate Bank evidence plus Qwen multimodal adjudication.

## D012 — Integration PASS has a narrow meaning

**Status: LOCKED.** PASS means pipeline execution, structural validity, resolver completion, and submission-contract validity. It does not mean competition accuracy.

## D013 — Development evaluation is the next quality gate

**Status: NEXT.** Run the complete pipeline on labelled development sequences, use `dataset/score.py`, preserve sequence boundaries, and publish an error taxonomy for text, order, speaker, identity, and serialization.

## D014 — Runtime optimization follows measurement

**Status: OPEN.** Add telemetry for model, stage, page, region/balloon, duration, success, and failure before changing the locked architecture. Qwen adjudication is currently the dominant explicitly measured stage.
