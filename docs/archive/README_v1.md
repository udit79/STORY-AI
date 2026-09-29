# STORY-AI — Manga Story Extraction & Speaker Resolution

STORY-AI is an automatic three-page manga understanding pipeline for the BYTE Applied ML Manga Task.

Given **three consecutive English manga pages**, the system extracts story text in reading order, determines who spoke each line where possible, and keeps character identities consistent across the three-page sequence.

The system is designed around local/open-weight inference and deterministic post-processing. Hosted AI inference is not part of the production target path.

## Task

For each three-page sequence, produce:

```json
{
  "sequence_id": "seq_...",
  "pages": [
    [
      {"speaker": "A", "text": "..."}
    ],
    [
      {"speaker": "NARRATION", "text": "..."}
    ],
    [
      {"speaker": "B", "text": "..."}
    ]
  ]
}
```

Character labels are sequence-local. `A`, `B`, `C`, ... are assigned only after cross-page identity resolution.

The system must automatically exclude non-story material such as SFX, credits, watermarks, page numbers, ads, and other task-excluded text according to the competition specification.

## Locked Architecture

```text
3-page sequence
      ↓
Page perception
      ↓
CTD text localization
      ↓
Panel/layout + balloon grouping
      ↓
Deterministic localized crops
      ↓
PaddleOCR + Nemotron + Qwen evidence
      ↓
Candidate Bank
      ↓
Qwen3-VL multimodal balloon adjudicator
      ↓
Story-text inclusion + semantic type
      ↓
Character perception
      ↓
Speaker grounding
      ↓
Reading order
      ↓
Cross-page character identity
      ↓
Sequence consistency resolver
      ↓
Validator
      ↓
Competition JSONL
```

**LangGraph** is the control plane around these stages. It manages sequence state, routing, retries, dependencies, parallel execution, and diagnostics.

### Architectural separation

```text
CTD / layout          = WHERE is relevant text/layout?
OCR models            = WHAT might the text say?
Qwen adjudicator      = WHAT does the image actually say?
Geometry + grounding  = WHO/WHERE does the utterance belong to?
Identity graph        = SAME CHARACTER across pages?
Resolver + validator  = WHAT survives into final JSONL?
```

## Real Components

| Function | Component | Status |
|---|---|---|
| Text localization | Comic Text Detector (`comictextdetector.pt.onnx`) | Integrated |
| Panel / balloon layout | `manga109-segmentation-bubble` YOLO | Integrated |
| OCR | PaddleOCR `en_PP-OCRv5_mobile_rec_onnx` | Integrated |
| OCR evidence | Nemotron OCR v2 via WSL worker | Integrated |
| Multimodal transcription | Qwen3-VL-4B-Instruct, 4-bit | Integrated |
| Character detection | RT-DETRv4 Manga109 model | Integrated |
| Character embeddings | MobileNetV3 | Integrated |
| Speaker visual grounding | Qwen3-VL | Integrated |
| Orchestration | LangGraph | Locked |

## Candidate Bank

The Candidate Bank deliberately retains multiple hypotheses instead of assuming that the highest-confidence OCR result is correct.

Sources can include:

```text
PaddleOCR
Nemotron
Qwen proposal evidence
```

Qwen adjudication is image-primary and may correct all proposals when they are jointly wrong.

## Speaker Labels

The serialization boundary enforces:

```text
NARRATION
  narration / thought / sound-effect / other non-speaker story text

UNKNOWN
  unresolved or ambiguous speaker / identity

A, B, C, ...
  sequence-resolved character identities
```

Ambiguous identities must not be converted into fabricated character labels.

## Current Integration Baseline

Real 3-page smoke sequence:

```text
seq_952f154fb1505883
```

Reported regression suite:

```text
212/212 tests passing
```

Real integration counts:

```text
pages:             3
CTD regions:       34
balloons:          29
story balloons:    29
candidates:       119
characters:        31
speaker resolved:  15
narration:         14
identity clusters: 12
identity pairs:   318
validation errors: 0
```

The end-to-end integration serialized successfully and passed the competition-format validator.

**Important:** integration `PASS` means execution and contract correctness. It does not mean that the model has achieved high competition accuracy. Quality is measured separately on the labelled development set.

## Runtime Baseline

The three-page integration measured approximately:

| Stage | Time |
|---|---:|
| CTD | 24.35 s |
| Layout / grouping | 1.34 s |
| Crop generation | 0.05 s |
| OCR + Qwen proposals | 353.94 s |
| Qwen adjudication | 1146.75 s |
| Character detection | 11.29 s |
| Reading order | 0.01 s |
| Character identity | 2.81 s |

Qwen adjudication is currently the largest explicitly instrumented stage.

The smoke runner should be further instrumented with per-call timing for:

```text
Qwen proposals
Qwen adjudication
Qwen visual speaker grounding
Nemotron
PaddleOCR
```

before performance-oriented architecture changes are made.

## Repository Layout

```text
STORY-AI/
├── app/
│   ├── schemas/
│   ├── models/
│   ├── langgraph_nodes/
│   ├── page_perception.py
│   ├── layout_grouping.py
│   ├── candidate_bank.py
│   ├── balloon_adjudication.py
│   ├── character_perception.py
│   ├── speaker_grounding.py
│   ├── reading_order.py
│   ├── character_identity.py
│   ├── sequence_resolver.py
│   ├── sequence_validator.py
│   └── submission_serializer.py
├── tools/
│   ├── pipeline_integration_smoke.py
│   ├── nemotron_worker.py
│   └── ...
├── tests/
├── vendor/
│   └── comic-text-detector/
├── dataset/
│   ├── development/
│   └── test/
├── .smoke/
├── pyproject.toml
├── README.md
├── STORY_AI_DEVELOPMENT_LOG.md
└── STORY_AI_DECISIONS.md
```

## Important Files

### Development log

`STORY_AI_DEVELOPMENT_LOG.md`

Chronological record of experiments, evidence, implementation milestones, failures, and reasoning.

### Decisions register

`STORY_AI_DECISIONS.md`

Stable architecture decisions. Changes should be recorded as a new decision rather than silently rewriting an old decision.

### Integration smoke

`tools/pipeline_integration_smoke.py`

Runs the full real 3-page development integration.

Typical entry point:

```powershell
.venv\Scripts\python.exe tools\pipeline_integration_smoke.py
```

Useful diagnostics:

```powershell
.venv\Scripts\python.exe tools\pipeline_integration_smoke.py --skip-visual-speaker
.venv\Scripts\python.exe tools\pipeline_integration_smoke.py --skip-nemotron
.venv\Scripts\python.exe tools\pipeline_integration_smoke.py --skip-qwen
```

## Nemotron WSL Bridge

Nemotron runs behind a persistent WSL worker to isolate its Linux-side environment.

```text
Windows
  ↓
NemotronWSLBridge
  ↓
WSL worker
  ↓
Nemotron OCR v2
```

The worker is started once per pipeline run and shut down during cleanup.

Key files:

```text
tools/nemotron_worker.py
app/models/nemotron_wsl_bridge.py
tests/test_nemotron_wsl_bridge.py
```

## Development Workflow

Do not optimize from intuition alone.

Use this sequence:

```text
implement one component
        ↓
unit / structural tests
        ↓
small real smoke
        ↓
full 3-page integration
        ↓
labelled development evaluation
        ↓
error analysis
        ↓
targeted change
        ↓
regression test
```

The **80 labelled development sequences** are the main quality-evaluation target. Sequence boundaries must remain intact for train/validation splitting.

The **15 test sequences** are reserved for final automatic prediction and must not be manually edited.

## Current Project State

```text
Architecture                 ✅ LOCKED
Core components              ✅ Integrated
Nemotron WSL bridge          ✅ Integrated
Qwen visual speaker path     ✅ Integrated
Regression suite             ✅ 212/212 reported
3-page real integration      ✅ PASS
Competition serialization    ✅ PASS
Development-set evaluation   ⏳ NEXT
Runtime instrumentation     ⏳ NEXT
Performance optimization     ⏳ AFTER INSTRUMENTATION
Final test prediction        ⏳ LATER
```

## Laya Status

Laya has been permanently removed from the locked production architecture.

```text
Laya dependency              ❌
Laya candidate selector      ❌
Candidate Bank               ✅
Qwen multimodal adjudicator  ✅
```

The reason is architectural: candidate selection on hard manga OCR cases needs access to image pixels. Qwen3-VL is therefore the image-primary adjudicator.

## Documentation

Maintain these three documents together:

```text
README.md
STORY_AI_DEVELOPMENT_LOG.md
STORY_AI_DECISIONS.md
```

When a major experiment or architecture change is completed, update the development log first and record the corresponding stable decision in the decisions register.
