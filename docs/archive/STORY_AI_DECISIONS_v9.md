# STORY-AI Manga Task — Decisions Register

> This file records architecture and experiment decisions that should remain stable unless new evidence explicitly requires a change.

## D001 — Use LangGraph as the control plane

**Status:** LOCKED

LangGraph is the orchestration layer for sequence state, page workers, branching, retries, and final resolution.

CrewAI is optional and is not an additional control plane in the locked design.

---

## D002 — CTD is the localization layer

**Status:** LOCKED

Comic Text Detector is responsible primarily for finding text regions/lines.

It is not the final OCR engine and its language metadata is not trusted as an English-text filter.

---

## D003 — PaddleOCR is the primary OCR recognizer

**Status:** LOCKED

PaddleOCR remains the main deterministic recognition path.

Alternative preprocessing actions can be used selectively as retries/candidate generators.

---

## D004 — Do not make preprocessing the main Laya task

**Status:** LOCKED FOR CURRENT DEVELOPMENT

The preprocessing oracle showed that useful preprocessing gains are sparse and that some difficult cases are caused by line grouping/order or recognition failure rather than image preprocessing.

Therefore Laya is not currently being trained to choose only between image preprocessing actions.

---

## D005 — Qwen3-VL is a visual evidence source

**Status:** LOCKED

Qwen3-VL-4B-Instruct is used locally for:

- visual transcription,
- OCR verification,
- text-type evidence,
- spatial/semantic evidence,
- later speaker/character evidence where useful.

It is not the sole final transcription authority.

---

## D006 — Evaluate VLM at the localized block/utterance level

**Status:** LOCKED

Whole-page free-form transcription is unsuitable for candidate-level OCR comparison because it can merge multiple regions into one output.

Individual-line evaluation also does not align reliably with the story-utterance target.

The benchmark unit is therefore a localized CTD block/utterance containing all relevant lines for that region.

---

## D007 — OCR and VLM are parallel evidence sources

**Status:** LOCKED

The core OCR-resolution architecture is:

```text
             CTD block
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
   PaddleOCR          Qwen3-VL
        ↓                 ↓
   OCR candidates     VLM candidate/evidence
        └────────┬────────┘
                 ↓
           Candidate Bank
                 ↓
               Laya
```

This is preferred over a simple sequential `OCR → VLM → final` pipeline.

---

## D008 — Laya selects among candidates using structured evidence

**Status:** LOCKED

Laya does not receive raw page pixels as its primary input.

It receives structured evidence such as:

- candidate text,
- OCR confidence,
- VLM evidence,
- geometry/features,
- contextual information,
- optional semantic/text-type signals.

The initial Laya task is candidate selection, not free-form OCR generation.

---

## D009 — Do not directly average OCR and VLM confidence

**Status:** LOCKED

OCR recognition scores and VLM self-reported visual confidence are not assumed to share a calibrated probability scale.

They are treated as features/evidence. A learned decision layer and later calibration determine how much each signal should influence the result.

---

## D010 — Preserve all candidate outputs until resolution

**Status:** LOCKED

The system should maintain a Candidate Bank rather than immediately collapsing to one transcription.

Candidate sources may include:

```text
OCR_BASE
OCR_UPSCALE
OCR_CONTRAST
OCR_THRESHOLD
VLM
FUSED
```

The exact active candidate set can be reduced later based on measured utility.

---

## D011 — Every experiment must be traceable to page and block

**Status:** LOCKED

Minimum provenance fields for OCR/VLM experiments:

```text
sequence_id
page_index
page_path
block_id
bbox
crop_path
```

Results must also preserve OCR/VLM outputs and comparison metrics.

---

## D012 — Do not treat the 20-line VLM result as a model-quality verdict

**Status:** LOCKED

The 15% exact OCR/VLM agreement observed in the line-level experiment is diagnostic only.

It is not a valid statement that either model is 85% wrong or 15% right because the experiment mixed line-level outputs with block/utterance-level visual reconstruction.

---

## D013 — Block-level benchmark comes before Laya training

**Status:** LOCKED

Before training Laya, measure on development data:

```text
CER(OCR, GT)
CER(VLM, GT)
CER(best candidate, GT)
```

and inspect cases where OCR and VLM disagree.

The next implementation task is the block-level benchmark.

---

## D014 — Sequence-level train/validation split

**Status:** LOCKED

Complete three-page sequences must remain intact when creating development train/validation splits. Individual pages or blocks must not leak across splits.

---

## D015 — Test inference remains fully automatic

**Status:** LOCKED

The final test pipeline must automatically produce the required JSONL output. No manual correction of test predictions is part of the intended system.

---

## Current next action

```text
Run block-level CTD → PaddleOCR vs Qwen3-VL benchmark
        ↓
measure CER at the correct unit
        ↓
inspect disagreement cases
        ↓
construct Candidate Bank dataset
        ↓
fine-tune/calibrate Laya
```


---

## D016 — Use CTD `bounding_rect()` for block geometry

**Status:** LOCKED

The installed CTD `TextBlock` implementation exposes `bounding_rect` as a callable method.

The benchmark must use:

```python
block.bounding_rect()
```

and convert:

```text
[x, y, width, height]
```

to:

```text
[x1, y1, x2, y2]
```

Do not access a nonexistent `bounding_box` attribute.

---

## D017 — OCR sanity gate before benchmark execution

**Status:** LOCKED

The block benchmark must perform a minimal PaddleOCR sanity check before processing benchmark blocks.

The sanity check must produce non-empty text and a valid recognition score. In the latest corrected run:

```text
HUH? DOES
0.9922
```

was returned, so the OCR environment was confirmed operational.

If the sanity check fails, the benchmark must stop instead of converting OCR failure into artificial CER=1.0 results.

---

## D018 — Oracle integrity is a prerequisite for benchmark metrics

**Status:** LOCKED

The benchmark must not compare OCR and VLM candidates unless the ground-truth correspondence layer is verified.

The latest run exposed:

```text
Oracle builder rows : 8
Oracle loader rows  : 2
```

with all processed blocks receiving `GT=None`.

Therefore this run's CER and candidate-winner metrics are invalid.

The benchmark must verify:

```text
builder row count == loader row count
```

and then verify block-to-GT correspondence before computing quality metrics.

---

## D019 — Block-level VLM inference is operational, but not yet a quality verdict

**Status:** CURRENT

Qwen3-VL successfully produced localized block transcriptions and text-type classifications, including dialogue and sound-effect examples.

However, because GT correspondence was broken in the latest run, these outputs are qualitative evidence only.

No OCR-vs-VLM quality conclusion should be drawn until the same GT row is attached deterministically to both candidates.

---

## D020 — Do not start Laya training until benchmark integrity passes

**Status:** LOCKED

Laya training remains blocked until:

```text
CTD localization
    ↓
block/line OCR candidate
    ↓
VLM block candidate
    ↓
same deterministic GT target
    ↓
validated CER / candidate labels
```

has been demonstrated on the small benchmark.

The latest run does not satisfy this condition.

---

## Current next action

```text
Fix oracle JSONL loader/keying
        ↓
validate all oracle rows are loaded
        ↓
validate CTD block → GT correspondence
        ↓
rerun seq_2032620aa4e4ac7f
        ↓
measure OCR vs VLM CER
        ↓
inspect disagreements
        ↓
construct Candidate Bank dataset
        ↓
fine-tune/calibrate Laya
```


---

## D021 — Oracle loader schema fixed

**Status:** RESOLVED

The oracle loader now uses the actual persisted field:

```python
block_index
```

instead of a nonexistent `block_id` field with a zero fallback.

The resulting benchmark run changed from:

```text
Oracle rows: 8
Loaded oracle rows: 2
```

to:

```text
Oracle rows: 8
Loaded oracle rows: 8
```

This confirms that the previous dictionary-key collision has been removed.

---

## D022 — Page/block correspondence remains a blocking integrity issue

**Status:** LOCKED

Loading all oracle rows is necessary but not sufficient.

The latest run attached the third-page oracle rows to runtime page 2 blocks, while the corresponding runtime page 3 blocks remained unmatched.

Therefore the benchmark still has a page-index / identity convention mismatch.

The benchmark must validate correspondence using page identity and block identity, not merely confirm that the oracle dictionary contains the expected row count.

---

## D023 — No CER aggregate until block-to-GT identity is verified

**Status:** LOCKED

The latest aggregate output:

```text
Reliable matches : 3
Mean OCR CER      : 0.8943
Mean VLM CER      : 1.4162
```

is invalid for model comparison because the attached GT rows are not verified to belong to the same page/block.

Therefore these values must not be used for:

```text
model selection
candidate labeling
Laya training
architecture claims
```

---

## D024 — Benchmark must validate page identity explicitly

**Status:** LOCKED

For every oracle lookup, the benchmark should preserve and validate:

```text
sequence_id
image/page path
page index
block index
bbox
```

The benchmark should fail before CER calculation when a GT row's page identity does not agree with the runtime block's page identity.

---

## Current next action

```text
Fix page-index / page-identity convention
        ↓
validate all 8 oracle correspondences
        ↓
rerun seq_2032620aa4e4ac7f
        ↓
obtain valid OCR vs VLM CER
        ↓
inspect complementary failure cases
        ↓
construct Candidate Bank dataset
        ↓
train/calibrate Laya
```

## D025 — First valid block-level OCR/VLM benchmark

**Status:** CONFIRMED FOR SINGLE-SEQUENCE DIAGNOSTIC

After fixing the oracle loader schema and the runtime/oracle page-index convention, `seq_2032620aa4e4ac7f` produced 8 verified oracle correspondences out of 13 processed CTD blocks.

Across the 8 matched blocks:

```text
Mean OCR CER : 0.3573
Mean VLM CER : 0.0047
```

VLM had lower CER on all 8 matched blocks in this sequence.

This result is valid for this sequence-level diagnostic only. It is not sufficient for a general model-selection claim.

## D026 — Do not replace the Candidate Bank with VLM-only selection

**Status:** LOCKED

The first valid sequence demonstrates strong VLM complementarity, but at least one VLM result still contains a transcription discrepancy (`EMBAR- RASSING` versus `EMBARRASSING`).

Candidate Bank remains the required architecture:

```text
OCR variants + VLM evidence
            ↓
       Candidate Bank
            ↓
           Laya
```

No hard rule of the form `always choose VLM` is permitted based on the single-sequence result.

## D027 — Story-text eligibility is a separate problem from transcription quality

**Status:** LOCKED

The latest benchmark includes visible text regions for which the development oracle contains no story utterance. Qwen3-VL can correctly transcribe such regions while still producing text that is not part of the benchmark target.

Therefore the system must distinguish:

```text
text recognition
story-text eligibility
```

This supports retaining VLM `text_type` and contextual/spatial evidence in the structured decision input.

## D028 — Expand benchmark across complete development sequences before Laya

**Status:** NEXT

Run the corrected benchmark on:

```text
seq_952f154fb1505883
seq_7db7a000adeceacf
seq_9032e5551cf1a9d2
```

Aggregate only after processing complete sequences so that sequence-level train/validation integrity is preserved.

## D029 — Laya dataset remains blocked until multi-sequence evidence

**Status:** LOCKED

Do not construct final Laya training labels from `seq_2032620aa4e4ac7f` alone.

The next dataset must contain, across multiple complete sequences:

```text
candidate source
candidate text
OCR recognition evidence
VLM evidence
geometry/features
story-text eligibility evidence
GT target
candidate CER
```

Only then should Laya candidate ranking/selection be trained or calibrated.

## Current next action

```text
Run corrected benchmark on 3–4 more sequences
        ↓
aggregate OCR/VLM/preprocessing behavior
        ↓
analyze story-text false positives
        ↓
build Candidate Bank examples
        ↓
split by complete sequence
        ↓
train/calibrate Laya
```

## D030 — Expanded four-sequence OCR/VLM diagnostic is informative but bounded

**Status:** CONFIRMED FOR DIAGNOSTIC USE

Four sampled development result files now cover 58 processed CTD blocks and 47 rows with attached GT.

```text
Mean OCR CER : 0.2508
Mean VLM CER : 0.0081

VLM lower CER : 45 / 47
OCR lower CER : 0 / 47
Equal         : 2 / 47
```

This is stronger evidence of Qwen3-VL complementarity than the earlier one-sequence run, but it remains a CTD-localized, limited-block diagnostic. The unmatched 11 processed rows cannot be interpreted as recall because the benchmark used a per-page processing limit.

## D031 — Oracle/runtime bbox verification must tolerate detector drift without hiding identity errors

**Status:** REQUIRED IMPLEMENTATION FIX

A rerun produced:

```text
page=1, block=0, IoU=0.836
```

The existing hard `0.85` assertion is too brittle to serve as the sole identity test for repeated CTD runs. The benchmark must separate:

```text
CTD deduplication threshold
oracle/runtime identity verification
bbox drift diagnostics
```

A near-threshold IoU should be reported for review rather than automatically treated as a block-identity failure. Low-overlap cases still require an explicit failure path.

## D032 — Nemotron must first be tested as a standalone full-page OCR system

**Status:** LOCKED

Before adding Nemotron to the Candidate Bank, test it without CTD, PaddleOCR, Qwen3-VL, preprocessing, or Laya.

Required first path:

```text
full manga page
      ↓
Nemotron OCR v2
      ↓
its own regions + text + confidence
```

The first evaluation should measure detection, recognition, grouping, and runtime separately. GT matching must be geometric, not region-index based.

## D033 — Use the English Nemotron v2 variant for the first manga benchmark

**Status:** LOCKED FOR FIRST RUN

The task pages are English. The first Nemotron run should therefore use:

```python
NemotronOCRV2(lang="en")
```

with `merge_level="sentence"` as the initial page-level grouping. A `word`-level pass is a follow-up diagnostic if sentence grouping is unsuitable for manga balloons. The official quickstart documents these entry points and merge levels. citeturn756805search0turn756805search8

## Current next action

```text
Resolve/revise bbox identity guard
        ↓
Run Nemotron full-page on seq_2032620aa4e4ac7f
        ↓
Save raw Nemotron regions
        ↓
Geometrically match to GT
        ↓
Compare detection + CER + grouping against CTD baseline
        ↓
Decide whether Nemotron contributes a useful independent OCR/layout signal
        ↓
Only then consider adding Nemotron to Candidate Bank
```

## D034 — Nemotron standalone benchmark uses raw word-level path as primary diagnostic

**Status:** LOCKED FOR FIRST QUANTITATIVE RUN

The first full-page manga probe showed that `merge_level="sentence"` and `merge_level="paragraph"` did not produce useful utterance-level grouping on `seq_2032620aa4e4ac7f/01.png`. The observed output remained fragmented into many regions for a single central dialogue balloon.

The benchmark therefore uses:

```python
NemotronOCRV2(model_dir=...)
ocr(page, merge_level="word")
```

with:

```python
NemotronOCRV2(skip_relational=True)
```

for the primary standalone capability measurement.

This isolates Nemotron's detector/recognizer behavior from its relational grouping stage.

## D035 — Nemotron quantitative evaluation must separate coverage from GT-conditioned recognition

**Status:** LOCKED

The standalone benchmark reports:

```text
GT story-block coverage
GT-conditioned recognition CER
GT-conditioned exact match rate
region counts
runtime
story-GT-box membership
```

GT-conditioned recognition is obtained by collecting Nemotron regions whose geometry falls inside the known oracle story block. It is explicitly **not** treated as end-to-end story extraction performance.

End-to-end story filtering/reading-order performance remains a later experiment.

## D036 — Nemotron is an independent OCR candidate, not the final story-text segmenter

**Status:** CONFIRMED BY PROBE

The page-01 probe showed:

```text
Nemotron independently detects dialogue text            ✅
Nemotron independently recognizes dialogue text         ✅
Nemotron detects non-story text as well                  ✅
Nemotron's raw output requires grouping                  ✅
Nemotron's sentence/paragraph grouping is not yet
sufficiently aligned with manga utterance blocks         ✅ observed
```

Therefore Nemotron should not replace CTD's manga-specific localization or the sequence-level story-text decision layer based on this evidence alone.

It remains a candidate independent OCR/layout evidence source for the Candidate Bank, pending the 3-page quantitative benchmark.

## Current next action

```text
Run nemotron_fullpage_gt_benchmark.py on seq_2032620aa4e4ac7f
        ↓
inspect 3-page raw regions + GT-conditioned metrics
        ↓
compare with previous CTD/PaddleOCR/Qwen evidence
        ↓
decide whether Nemotron contributes independent candidate/layout evidence
```

## D037 — Nemotron standalone 3-page benchmark completed

**Status:** COMPLETED / DIAGNOSTIC

Sequence:
`seq_2032620aa4e4ac7f`

Configuration:
- `nvidia/nemotron-ocr-v2`
- English checkpoint
- full-page input
- `merge_level="word"`
- `skip_relational=True`
- no CTD, PaddleOCR, Qwen3-VL, preprocessing, or Laya

Observed metrics:

```text
GT story blocks                  : 8
GT story blocks covered          : 8 / 8
GT-conditioned mean CER         : 0.1990
GT-conditioned exact-match rate : 25%
Total predicted regions         : 142
Inside story GT boxes           : 67
Outside story GT boxes          : 75
```

The recognition metrics are explicitly GT-conditioned and are not end-to-end story extraction metrics.

## D038 — Nemotron demonstrates useful independent OCR evidence

**Status:** CONFIRMED FOR FURTHER VALIDATION

On the benchmark sequence, Nemotron geometrically covered all 8 labelled story blocks. It also recognized two blocks exactly and recovered substantial text content in the remaining blocks.

This is sufficient evidence to keep Nemotron in the experimental Candidate Bank evaluation set rather than discarding it after the qualitative probe.

## D039 — Nemotron remains unsuitable as the final story utterance segmenter on current evidence

**Status:** LOCKED

The page-01 word/sentence/paragraph probes and the standalone raw-word benchmark showed that Nemotron's output is fragmented relative to the benchmark's utterance/block unit. The model also emits substantial non-story text.

Therefore:

```text
Nemotron = OCR evidence expert
Nemotron ≠ final story-text filter
Nemotron ≠ final utterance grouping
Nemotron ≠ final reading-order resolver
```

Those responsibilities remain with the downstream story-text, reading-order, speaker-grounding, and sequence-consistency layers.

## D040 — Do not infer steady-state Nemotron throughput from page-01 runtime

**Status:** LOCKED

Observed runtimes:

```text
page 01 : 22,927.6 ms
page 02 :    402.1 ms
page 03 :    219.5 ms
```

Page 01 includes model initialization/cold-start effects. Report cold-start and steady-state runtime separately in future benchmarks.

## D041 — Candidate Bank expansion now requires cross-sequence validation

**Status:** NEXT GATE

Nemotron remains an experimental Candidate Bank source after the first quantitative run, but its inclusion in the locked production path is not yet justified from one sequence.

Next evaluation should use additional labelled development sequences and the same protocol:

```text
full page
   ↓
Nemotron
   ↓
geometric GT conditioning
   ↓
coverage + CER + exact match + false-positive region statistics + runtime
```

Only after cross-sequence validation should Nemotron's role in the final Candidate Bank be locked.

## D042 — Nemotron approved as a Candidate Bank evidence source

**Status:** LOCKED

Four labelled development sequences were evaluated with the same standalone Nemotron configuration (`merge_level=word`, `skip_relational=true`). The benchmark covered 91/91 GT story blocks geometrically, with block-weighted mean GT-conditioned CER of 0.2375 and 14/91 exact matches. The result is consistent enough across the four sequences to justify retaining Nemotron as an independent OCR evidence expert in the Candidate Bank.

This approval is specifically for **evidence generation**, not for final story reconstruction.

## D043 — Nemotron scope remains bounded downstream

**Status:** LOCKED

Nemotron will not be used as the final authority for story-text filtering, utterance grouping, reading order, speaker attribution, or cross-page character identity. Those decisions remain downstream and must combine multiple evidence sources plus sequence-level context.

Nemotron confidence must not be treated as a calibrated probability without held-out calibration.

## D044 — Candidate Bank becomes the next implementation boundary

**Status:** NEXT GATE

Implement a typed Candidate Bank / evidence fusion layer that preserves multiple hypotheses per localized CTD block before Laya selection.

Minimum sources:

```text
CTD-localized block
   ├── PaddleOCR candidates
   ├── Nemotron candidates/evidence
   └── Qwen3-VL candidate/evidence
           ↓
      Candidate Bank
           ↓
          Laya
```

The bank must retain candidate provenance, text, confidence/evidence features, geometry, preprocessing/action identity, and source-specific metadata without collapsing them into a single averaged confidence.

## D045 — Laya is removed from the production architecture

**Status:** LOCKED / SUPERSEDES D044's Laya-selection path

The architecture no longer uses Laya as the Candidate Bank decision layer.
The previous Laya design was image-free and restricted the policy to selecting one supplied candidate ID. The revised design requires the final transcription decision to have access to the actual localized balloon image because all OCR candidates may be jointly wrong.

Laya-specific production components are therefore removed from the final path:

```text
CandidateBank → Laya → selected candidate
```

is superseded by:

```text
CandidateBank + balloon image + layout context
                    ↓
             multimodal adjudicator
                    ↓
          final balloon transcription
```

The existing Laya code may remain temporarily for historical/reference purposes during migration, but it is not a required runtime dependency.

## D046 — Balloon/utterance is the semantic OCR unit; CTD region is not

**Status:** LOCKED

A CTD text region is a localization/provenance unit, not necessarily a complete semantic utterance. Multiple CTD regions may belong to one speech/thought balloon or caption, and a balloon may contain multiple text fragments.

Required structure:

```text
CTD regions
    ↓
balloon / caption / utterance grouping
    ↓
balloon-level evidence
```

CTD region IDs remain authoritative provenance identifiers. Balloon IDs are downstream semantic grouping identifiers and must never overwrite the CTD provenance relationship.

This design is motivated by recent manga annotation research documenting missing text regions, overlapping dialogue/onomatopoeia, and under-segmented speech balloons.

## D047 — Qwen3-VL becomes the multimodal transcription adjudicator

**Status:** LOCKED

Qwen3-VL is used as the final localized transcription adjudicator. It receives:

- the actual balloon/utterance image crop;
- PaddleOCR candidate text and evidence;
- Nemotron candidate text and evidence;
- optional earlier Qwen proposal;
- candidate geometry/spatial evidence;
- panel/context metadata where available.

The adjudicator must treat candidates as hypotheses, not ground truth, and must be allowed to produce a corrected transcription not present verbatim in any candidate.

The adjudicator output must include at minimum:

```text
final_text
text_type
include_in_story
confidence / uncertainty metadata
```

The adjudicator must not paraphrase or normalize visible wording.

## D048 — Visual evidence has priority over candidate agreement

**Status:** LOCKED

Candidate agreement is supporting evidence, not a voting mechanism.

The adjudicator policy is:

```text
image = primary evidence
candidates = hypotheses
```

It must not use majority voting or highest-confidence-wins as the final transcription rule.

This specifically addresses cases where PaddleOCR, Nemotron and Qwen may share a systematic glyph/word error.

## D049 — Panel/layout structure is an auxiliary context layer

**Status:** LOCKED

Introduce an auxiliary layout representation containing panels or equivalent page regions when confidently available.

Required hierarchy:

```text
page
 ├── panel(s)
 │    ├── balloon(s)
 │    └── character(s)
```

Panel/layout detection is not allowed to become a single point of failure. When layout confidence is insufficient, the system falls back to page-coordinate geometry and downstream evidence.

## D050 — Story-text filtering is explicit and multimodal

**Status:** LOCKED

The system must explicitly distinguish story-bearing text from excluded text such as SFX, signs, titles, metadata, document/interface text, and other non-story regions according to the task specification.

The adjudication/evidence layer may classify:

```text
dialogue
thought
narration
vocalisation
sound_effect
sign
title
metadata
document_text
unknown
```

and provide an `include_in_story` decision, but final validation must retain the ability to reject inconsistent outputs.

Visual presentation is part of the evidence because identical-looking strings may represent dialogue, SFX, signage, or other excluded content.

## D051 — Reading order is hierarchical and panel-aware

**Status:** LOCKED

Final reading order is not delegated to a single generative LLM and is not implemented as a global x/y sort.

The intended hierarchy is:

```text
page order
  ↓
panel order
  ↓
balloon order within panel
  ↓
text order within balloon
```

A lightweight learned pairwise ordering model may be trained on the 80 development sequences using geometry/layout features. This provides a clear task-adapted learned component without requiring full VLM fine-tuning.

Reading direction must be learned/verified from development labels and not assumed solely from language.

## D052 — Speaker grounding is a multimodal relation problem

**Status:** LOCKED

Speaker attribution is not inferred from text alone.

For each balloon, generate candidate speakers from characters in the relevant panel/scene and combine:

- balloon-tail geometry when present;
- spatial proximity;
- panel membership;
- character visibility;
- character visual evidence;
- balloon/image context.

The resolver must allow `unknown` internally when evidence is insufficient and must preserve the evidence used for the final assignment.

Research on Manga109Dialog supports scene-graph and reading-order-aware speaker detection rather than distance-only assignment.

## D053 — Cross-page character consistency is a sequence-local identity graph

**Status:** LOCKED

Character labels are arbitrary within each three-page sequence. Therefore the identity problem is sequence-local rather than global character-name recognition.

Pipeline:

```text
page characters
      ↓
pairwise cross-page visual similarity
      ↓
identity graph
      ↓
identity clusters
      ↓
anonymous sequence labels
```

Evidence may include face/body appearance, clothing/hair/style, contextual location, embeddings, and VLM verification. The resolver must preserve one label for the same character across pages and distinct labels for different characters.

## D054 — Sequence-level consistency is a separate global resolver

**Status:** LOCKED

After local transcription, filtering, order, speaker, and identity predictions are available, a sequence-level resolver reconciles them under explicit constraints.

It must enforce at least:

```text
valid source/provenance references
acyclic reading order
valid speaker references
consistent anonymous character identities
no excluded text in final story output
no invented source regions
```

This layer should be deterministic or optimization-based where possible, rather than a free-form generative LLM.

## D055 — Learned-component strategy is hybrid, not LLM-only

**Status:** LOCKED

The system will harness Qwen3-VL as the multimodal adjudicator and visual reasoning component.

For an explicit task-adapted learned component, the preferred first training target is a lightweight pairwise reading-order ranker trained on development sequences. This is deliberately narrower than fine-tuning the 4B VLM and is intended to reduce complexity and provide an auditable learned decision boundary.

Full Qwen fine-tuning is not part of the initial architecture and requires evidence from development errors before being considered.

## D056 — No further standalone OCR experiments are required at this stage

**Status:** LOCKED

The four-sequence Nemotron validation is sufficient to retain Nemotron as Candidate Bank evidence. Additional OCR benchmarks are not a current development gate.

Future OCR experiments are allowed only when they answer a concrete integration regression or a failure observed in the development pipeline.

## D057 — Final production architecture

**Status:** LOCKED

The final architecture is:

```text
                    3 CONSECUTIVE PAGES
                              │
                              ▼
                    PAGE / LAYOUT PARSE
                         panels/context
                              │
                              ▼
                             CTD
                              │
                              ▼
                    TextRegion localization
                              │
                              ▼
                 BALLOON / UTTERANCE GROUPING
                              │
                       balloon crops
                              │
                  ┌───────────┼───────────┐
                  ▼           ▼           ▼
               Paddle      Nemotron    Qwen proposal
                  │           │           │
                  └───────────┼───────────┘
                              ▼
                       CANDIDATE BANK
                              │
                              │ + image
                              │ + panel context
                              ▼
                   MULTIMODAL ADJUDICATOR
                         Qwen3-VL
                              │
                              ▼
                    FINAL BALLOON TEXT
                              │
                              ▼
                     STORY-TEXT FILTER
                              │
                              ▼
                 PANEL-AWARE READING ORDER
                              │
                     learned pairwise ranker
                              │
                              ▼
                    CHARACTER PERCEPTION
                              │
                              ▼
                     SPEAKER GROUNDING
                              │
                              ▼
                  CROSS-PAGE IDENTITY GRAPH
                              │
                              ▼
                  SEQUENCE CONSISTENCY
                              │
                              ▼
                         VALIDATOR
                              │
                              ▼
                            JSONL
```

LangGraph remains the control plane around these components for parallel page execution, retries, dependency/state management, diagnostics, and deterministic routing. It is orchestration, not a substitute for the component-level models.

## D058 — Architecture gap ownership

**Status:** LOCKED

Known gaps and their owners are:

| Gap | Resolution owner |
|---|---|
| CTD fragment ≠ semantic utterance | Balloon grouping |
| SFX / story / sign ambiguity | Multimodal adjudicator + story filter |
| Complex layout | Layout layer + geometry fallback |
| Reading order | Hierarchical ordering + pairwise ranker |
| Speaker attribution | Tail/spatial/visual grounding |
| Cross-page character consistency | Sequence-local identity graph |
| LLM hallucination/paraphrase | Image-primary adjudication + validator |
| All candidates jointly wrong | Adjudicator may synthesize corrected text |
| Model/backend failures | LangGraph retries + page-perception diagnostics |
| Learned-component requirement | Pairwise order ranker + Qwen harness |
| Windows/WSL Nemotron boundary | Explicit injected local bridge |

## D059 — Implementation order after architecture lock

**Status:** NEXT GATE

Implementation order is fixed to reduce downstream rework:

```text
1. Balloon / utterance grouping + panel structure
2. Balloon-level evidence contract
3. Multimodal Qwen adjudicator
4. Story/non-story validation
5. Reading-order representation + ranker dataset
6. Character perception
7. Speaker grounding
8. Cross-page identity graph
9. Sequence consistency resolver
10. Validator + final JSONL
11. End-to-end development evaluation
```

The Nemotron WSL bridge may be implemented in parallel where required for Candidate Bank completeness, but it is not a reason to postpone the semantic-structure work.

---

## D060 — Full real 3-page integration baseline

**Status:** LOCKED

The complete locked architecture has been exercised on a real three-page development sequence:

```text
seq_952f154fb1505883
```

Reported regression suite:

```text
212/212 tests passing
```

The integration produced a valid competition-format JSONL output and passed validation with zero errors.

The integration baseline is therefore the reference point for subsequent development-set evaluation and performance work.

---

## D061 — All production backends integrated

**Status:** LOCKED

The current integration baseline successfully loads and uses:

```text
CTD
MangaLayout YOLO
PaddleOCR
Qwen3-VL 4B-Instruct
Nemotron WSL bridge
RT-DETRv4 character detection
MobileNetV3 embeddings
Qwen visual speaker grounding
```

No unavailable backend was reported in the final integration run.

---

## D062 — Nemotron uses a persistent WSL worker

**Status:** LOCKED

Nemotron is accessed through:

```text
Windows application
    ↓
NemotronWSLBridge
    ↓
persistent WSL worker
    ↓
Nemotron OCR v2
```

The model is loaded once per pipeline run rather than once per page or balloon.

The bridge is responsible for lifecycle management, request routing, and Windows↔WSL path conversion.

---

## D063 — Qwen remains the image-primary balloon adjudicator

**Status:** LOCKED

Qwen3-VL is shared by the proposal path, balloon adjudication path, and visual speaker-grounding path.

The semantic decision unit is the **balloon**, not an individual CTD fragment.

The adjudicator receives the balloon crop and candidate evidence and may produce corrected text even when every candidate is wrong.

Candidate Bank remains the multi-hypothesis evidence layer.

---

## D064 — Integration baseline does not imply task-quality success

**Status:** LOCKED

`PASS` in the integration smoke means:

```text
pipeline executes
+
schemas are valid
+
resolver completes
+
submission contract is valid
+
validator reports zero errors
```

It does **not** establish competitive extraction accuracy.

Quality must now be measured against the labelled development set with the official scorer.

---

## D065 — Runtime optimization is a separate gate

**Status:** OPEN

The final smoke measured approximately:

```text
Qwen proposal production:  353.94 s
Qwen adjudication:       1146.75 s
```

Qwen adjudication is the largest explicitly instrumented stage.

However, speaker grounding currently lacks a dedicated runtime entry, so the wall-clock total is not completely explained by the stage report.

Before deleting or restructuring components, instrument per-model call counts and timings.

Required runtime telemetry:

```text
model
stage
page
balloon / region id
start time
duration
success/failure
```

Performance changes must preserve the locked semantic architecture and be justified by measured evidence.

---

## D066 — Next evaluation gate is the labelled development set

**Status:** NEXT GATE

The next decision cycle is:

```text
80 labelled development sequences
        ↓
automatic inference
        ↓
official score.py
        ↓
text / order / speaker metrics
        ↓
error taxonomy
        ↓
targeted model or rule improvements
```

The 15 held-out test sequences remain untouched until the production pipeline and development evaluation are finalized.

