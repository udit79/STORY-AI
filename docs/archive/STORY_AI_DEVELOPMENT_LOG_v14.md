# STORY-AI Manga Task — Progressive Development Log

> **Purpose:** Keep a chronological, reasoning-first record of how the system is being developed.  This document records not only what was implemented, but also the problem we were trying to solve, the experiment we ran, what the evidence showed, and why the next design decision followed.
>
> **Working principle:** Do not force an architecture because it sounds good. Run a small experiment, inspect the evidence, identify the actual bottleneck, then change one part of the system and test again.

---

## 0. Project objective

The project takes **three consecutive English manga pages** and must automatically:

1. extract story text in reading order,
2. identify who spoke each line,
3. keep character labels consistent across the three pages,
4. produce the required JSONL output automatically.

The development set contains labelled three-page sequences; the test set contains unlabelled sequences. Sequence boundaries must remain intact when splitting development data for training/validation.

The task is not only OCR. The final system must distinguish story text from SFX and other excluded text, determine reading order, ground text to speakers, and resolve character identity across pages.

---

# 1. Initial architecture and the first problem framing

## 1.1 Initial idea

The initial architecture was designed as a multi-stage system:

```text
3-page sequence
      ↓
parallel page perception
      ↓
OCR + VLM + text/balloon localization
      ↓
structured page representation
      ↓
page fusion
      ↓
sequence-level resolver
      ↓
story-text filtering
reading order
speaker grounding
cross-page character identity
      ↓
validation
      ↓
JSONL
```

LangGraph was selected as the orchestration/control plane. The important architectural principle was to keep the perception layer separate from the reasoning/resolution layer.

## 1.2 Early realization

The system has several different failure modes. A generic “use a bigger model” approach would make it difficult to understand which component is actually failing.

We therefore moved toward **structured intermediate representations** and small experiments for each layer.

Mermaid was treated as a debugging/visualization aid rather than the source of truth.

---

# 2. Layer 1 — Build the structured representation and verify the data plumbing

## 2.1 What was implemented

The project was organized around explicit schemas for:

- `TextRegion`
- `CharacterInstance`
- `Balloon`
- `PageRepresentation`
- `SequenceRecord`
- `SequenceRepresentation`

The data layer was also created for loading `sequences.json`, resolving the three page paths for each sequence, and keeping development/test boundaries explicit.

## 2.2 Why this came first

Before testing models, we needed a stable contract describing what each page worker produces. This prevents later components from depending directly on model-specific outputs.

## 2.3 Result

The basic test suite passed, including the data/loader layer. This established a working project skeleton before model experiments were added.

**Decision:** Continue to the perception layer rather than adding more orchestration complexity.

---

# 3. CTD experiment — Can we reliably localize manga text?

## 3.1 Hypothesis

Comic Text Detector (CTD) was tested as the first text-localization component.

The question was deliberately narrow:

> **Can CTD locate the actual text regions in these manga pages well enough for us to crop them and hand them to an OCR recognizer?**

This separates localization from recognition.

## 3.2 Setup

CTD was run locally using the available ONNX model on the project GPU environment.

The detector returned text blocks containing individual text lines/polygons.

## 3.3 First observation

On the initial sequence, CTD successfully localized the visible target text regions on page 1. This was visually confirmed from the detector preview.

However, CTD also produced duplicate/near-duplicate blocks and regions corresponding to non-target text.

Example page-1 behavior included almost-identical horizontal and vertical detections for the same region.

## 3.4 Important discovery

CTD's language metadata was not reliable enough to serve as an English filter. English manga text was sometimes tagged with another language value.

Therefore:

> **Do not use CTD language classification as the primary story-text filter.**

## 3.5 Decision

CTD is retained primarily as a **localizer**, not as an OCR recognizer and not as the final semantic text classifier.

The next experiment was therefore OCR on CTD crops.

---

# 4. OCR experiment — CTD localization + PaddleOCR recognition

## 4.1 Hypothesis

If CTD can localize the text accurately, then PaddleOCR can recognize the cropped English text.

The first recognition experiment used CTD line crops directly.

## 4.2 First result

The raw line crops produced many recognition errors, such as misspellings and fragmented text. The errors were substantial enough that raw crops were not considered a sufficient OCR pipeline.

## 4.3 Why we did not immediately replace the OCR model

The detector polygons were sometimes tilted and the manga text could be small. Therefore the first question was whether **geometry and crop quality**, rather than OCR model capacity, were causing the errors.

This led to a preprocessing experiment.

---

# 5. OCR preprocessing experiment — rectify + upscale + padding + rotation

## 5.1 Changes tested

Before recognition, the CTD polygons were:

1. perspective-rectified,
2. upscaled,
3. optionally rotated for vertical text,
4. padded with white pixels.

## 5.2 Result on page 1

Recognition improved substantially on horizontal English text.

Representative behavior included high-confidence recognition of fragments such as:

```text
HUH? DOES
KOMABA
DIFFERENT
THAN USUAL?
HE'S
DOING
WORKING
HARD FOR
TOURNAMENT
NO
THAT'S
NOT IT.
OUT
FOR
BLOOD.
```

Some difficult/vertical regions still produced poor results.

## 5.3 Key interpretation

This experiment established an important separation:

```text
CTD localization          → generally useful
crop geometry              → materially affects OCR
PaddleOCR recognition      → strong on many horizontal regions
vertical/tiny cases        → still difficult
```

The next step was to test the complete three-page sequence rather than one page.

---

# 6. Three-page benchmark — Understand the real failure modes

## 6.1 What was tested

The same CTD + geometry preprocessing + PaddleOCR pipeline was run over all three pages of one development sequence.

## 6.2 Observed behavior

Page 1 contained several strong OCR results but also difficult regions.

Page 2 contained CTD detections that were not target story utterances, including an SFX-like region and other uncertain/garbled detections. The ground-truth page contained no story utterances for this sequence.

Page 3 contained several very strong OCR regions plus a difficult small region (`SO CLOSE!!`) and an SFX region (`CLACK`).

## 6.3 Main lesson

The system does not have one single “OCR problem.” The failure modes already appeared to include:

```text
localization
line segmentation
line reading order
recognition quality
SFX / non-story text
small or difficult text
speaker grounding
character identity
```

This distinction became important later when considering whether a learned preprocessing controller was actually solving the correct problem.

---

# 7. Laya research — Can a learned decision model choose preprocessing?

## 7.1 Idea

Instead of always applying expensive preprocessing, we considered a learned policy that would choose among alternatives such as:

```text
NONE
UPSCALE
CONTRAST
THRESHOLD
ROTATE_CW
ROTATE_CCW
```

The role proposed for Laya was **decision-making**, not image understanding or OCR itself.

## 7.2 Why Laya was considered

Laya is a typed-decision model intended to map structured state/evidence to a discrete choice. That matches a policy-selector role better than using it as a vision model.

## 7.3 Local setup

A separate uv-managed environment was created for the Laya policy so that the OCR environment would remain isolated.

The Laya checkpoint loaded successfully on the local GPU.

## 7.4 Critical result from the smoke test

The zero-shot model returned a choice such as `upscale`, but the probabilities were relatively flat and the checkpoint emitted a calibration warning. Therefore the zero-shot decision should **not** be treated as a reliable learned preprocessing policy.

## 7.5 Decision

Do not use the zero-shot Laya prediction to control OCR.

Instead, create a supervised **oracle dataset** from development ground truth and later fine-tune Laya on that decision task.

---

# 8. Oracle dataset — What it means in this project

## 8.1 Simple definition

The oracle dataset is a collection of examples where we already know the correct text because the development labels are available.

For each CTD region, we try several preprocessing actions, run OCR on each version, compare each OCR result with the known target text, and identify which action gave the lowest character error rate (CER).

Conceptually:

```text
known target text
      ↓
try NONE / UPSCALE / CONTRAST / ...
      ↓
run OCR for every action
      ↓
calculate CER against known target
      ↓
choose the best action
      ↓
training example for Laya
```

At test time, the target text is unavailable. Laya must predict the action from observable features and first-pass OCR evidence.

## 8.2 Why an oracle is allowed during development

The oracle is a **training-data construction mechanism**. It is not part of the final test-time system because the test set has no ground-truth text available during inference.

---

# 9. First oracle generator — initial result and problems found

## 9.1 First implementation

A prototype oracle generator was created for one development sequence.

It:

- ran CTD,
- recognized CTD lines,
- matched the resulting block text to ground-truth utterances,
- evaluated multiple preprocessing candidates,
- computed CER,
- selected the lowest-CER action.

## 9.2 First run

The first run produced 8 matched oracle rows.

Several rows had very high-quality OCR, and some preprocessing candidates were worse than the baseline.

However, we identified implementation issues in the oracle itself.

## 9.3 Problems identified

### Problem A — `NONE` was not actually NONE

The first preprocessing function automatically upscaled and automatically rotated some vertical crops even when the action was named `NONE`.

This made the experiment logically invalid.

### Problem B — the baseline was already preprocessed

The prototype used an upscaled baseline while also evaluating upscaling as a candidate. That prevents a clean comparison.

### Problem C — duplicate CTD blocks

CTD could return near-identical boxes for the same content. Without deduplication, the same ground-truth utterance could be represented multiple times.

### Problem D — greedy GT matching

Each block independently selected its best GT utterance. Two blocks could therefore claim the same GT item.

### Problem E — tiny/suspicious regions could be lost

Unmatched CTD regions still contained useful diagnostic information, but they were not the same thing as valid preprocessing training examples.

## 9.4 Decision

Before generating oracle data at scale, fix the oracle generator itself.

---

# 10. Corrected oracle generator — second implementation

## 10.1 Changes made

The oracle generator was rewritten so that preprocessing actions were explicit:

```text
NONE
UPSCALE
CONTRAST
THRESHOLD
ROTATE_CW
ROTATE_CCW
```

`NONE` became a true no-op after geometric rectification.

There was no hidden automatic rotation.

The baseline became the true `NONE` pipeline.

Near-duplicate CTD blocks were removed before matching.

Ground-truth matching became one-to-one.

Useful geometric/recognition features were added for eventual policy learning.

Unmatched regions were saved separately as diagnostics rather than being forced into the preprocessing oracle.

## 10.2 Result

The corrected generator ran successfully.

For the tested sequence:

```text
PAGE 1
Raw CTD blocks: 7
After deduplication: 5
GT utterances: 5

PAGE 2
Raw CTD blocks: 3
After deduplication: 3
GT utterances: 0

PAGE 3
Raw CTD blocks: 6
After deduplication: 6
GT utterances: 6

Oracle rows: 8
Unmatched rows: 6
```

The implementation was therefore operational and logically cleaner.

---

# 11. Candidate-by-candidate oracle inspection — the important discovery

## 11.1 Why we inspected individual candidates

The corrected run surprisingly produced `NONE` as the oracle action for every matched row in the sequence.

That could have meant either:

1. the oracle implementation was still wrong, or
2. preprocessing genuinely was not helping these examples.

Therefore, instead of changing the system immediately, we inspected every candidate OCR result.

## 11.2 What the detailed results showed

### Example: already-good OCR

Target:

```text
HUH? DOES KOMABA SEEM DIFFERENT THAN USUAL?
```

Baseline:

```text
HUH? DOES KOMABA SEEN DIFFERENT THAN USUAL?
```

CER was very low. Upscaling did not improve it, while contrast and thresholding were worse. Rotation was catastrophically wrong.

Interpretation: preprocessing is unnecessary for this region.

### Example: apparently poor block CER

Target:

```text
HE'S DOING REALLY WELL. IT LOOKS LIKE HE'S BEEN WORKING HARD FOR THE FALL TOURNAMENT.
```

Baseline OCR was approximately:

```text
HE'S DOING IT LOOKSREALLY LIKE HE'S WELL BEEN WORKINGHARD FOR THE FALL TOURNAMENT
```

CER was about `0.282`.

Crucially, the individual words/lines were not simply unreadable. The larger problem was how recognized pieces were ordered/grouped.

Interpretation: this is substantially a **line ordering / segmentation** problem, not something that contrast or upscaling alone will fix.

### Example: tiny/difficult text

Target:

```text
SO CLOSE!!
```

Baseline:

```text
CLOSE E
```

CER was `0.500`.

Upscaling, contrast, thresholding, and rotation all made the result worse.

Interpretation: the current preprocessing action set does not solve this recognition failure.

## 11.3 Main conclusion

The experiment changed our understanding of the problem.

We originally assumed:

```text
bad OCR
   ↓
preprocess better
   ↓
better OCR
```

The evidence shows a more complicated picture:

```text
CTD
 │
 ├── localization
 ├── line segmentation
 ├── line order
 ├── recognition quality
 ├── non-story text / SFX
 └── later semantic resolution
```

Therefore, a preprocessing-only policy may not provide enough value to justify making it the primary Laya task.

---

# 12. Current reasoning about Laya's role

## 12.1 Previous proposed role

```text
CTD crop
   ↓
Laya chooses image preprocessing
   ↓
OCR
```

## 12.2 Current hypothesis

Use Laya as a broader **OCR strategy selector** rather than assuming every difficult OCR case is a preprocessing problem.

Potential strategy choices could eventually include:

```text
DIRECT
REORDER_LINES
RECOGNIZE_LINES
RECOGNIZE_BLOCK
RETRY_TINY
```

These are conceptual candidates at this stage, not yet a locked label set.

## 12.3 Why this change is being considered

The latest oracle inspection indicates that at least some high-CER examples come from text ordering/grouping rather than image quality.

We should therefore test whether preprocessing is useful across a broader sample before committing Laya to that exact task.

---

# 13. Current project state

```text
Data / schemas                 ✅
Sequence-aware loader          ✅
CTD localization               ✅
OCR recognition                ✅
Basic geometry preprocessing   ✅
Three-page benchmark           ✅
Laya local inference           ✅
Oracle generator               ✅
Oracle logic corrected         ✅
Detailed oracle inspection     ✅

Preprocessing as Laya task     ⏳ not yet validated
Final Laya fine-tuning         ⏸️ not started
Story-text classifier          ⏸️ not started
Reading-order resolver         ⏸️ not started
Speaker grounding              ⏸️ not started
Character identity resolution  ⏸️ not started
Final automatic test pipeline  ⏸️ not started
```

---

# 14. Next experiment

Do **not** generate oracle data for the entire development set yet.

Run the corrected oracle generator on a small additional sample of development sequences.

The purpose is to measure the distribution of oracle actions:

```text
NONE
UPSCALE
CONTRAST
THRESHOLD
ROTATE_CW
ROTATE_CCW
```

Questions to answer:

1. Does preprocessing ever produce meaningful CER improvements across different sequences?
2. Which transformations actually help?
3. How often are difficult cases caused by recognition quality versus line order/segmentation?
4. Is preprocessing useful enough to be a worthwhile learned decision problem for Laya?

Only after this experiment should we decide between:

```text
A. keep Laya as a preprocessing policy,
```

or

```text
B. move Laya to a broader OCR/reading strategy decision layer.
```

---

# 15. Working methodology going forward

Every major change should be recorded using this chain:

```text
PROBLEM
  ↓
HYPOTHESIS
  ↓
SMALL EXPERIMENT
  ↓
OBSERVED RESULT
  ↓
INTERPRETATION
  ↓
DECISION
  ↓
NEXT EXPERIMENT
```

The goal is to preserve the reasoning trail so that later we can explain not only the final architecture, but also why alternative approaches were tested, rejected, or retained.

---

# 16. Decision log

| Date | Decision / event | Evidence | Consequence |
|---|---|---|---|
| 2026-09-28 | Use CTD as text localizer | CTD successfully localized target text but produced duplicates and non-target regions | Keep localization separate from recognition and semantic filtering |
| 2026-09-28 | Use PaddleOCR for first local OCR baseline | Horizontal English recognition became strong after geometry preprocessing | Continue with CTD → crop → OCR pipeline |
| 2026-09-28 | Do not trust CTD language field as English filter | English regions received unreliable language labels | Story-text classification must use stronger evidence |
| 2026-09-28 | Keep Laya in a separate environment | Laya is a decision model and should not contaminate OCR dependencies | Maintain isolated uv environments |
| 2026-09-28 | Do not use zero-shot Laya for OCR control | Smoke test showed weak/near-flat choice probabilities and calibration warning | Supervised fine-tuning required |
| 2026-09-28 | Build an oracle dataset from development labels | Ground truth can identify the best preprocessing action retrospectively | Enables supervised policy learning |
| 2026-09-28 | Correct oracle `NONE` and baseline logic | Prototype had hidden scaling/rotation | Oracle comparisons became valid |
| 2026-09-28 | Deduplicate CTD blocks and use one-to-one GT matching | Duplicate boxes and greedy matching could corrupt training labels | Cleaner oracle dataset |
| 2026-09-28 | Do not train Laya on first 8 oracle rows | All matched examples selected `NONE` | Need broader evidence before defining Laya's final task |
| 2026-09-28 | Investigate reading-order/segmentation failures | Some high-CER examples had mostly recognizable words but incorrect sequence/order | Preprocessing may not be the main decision problem |

---

# 17. Important implementation notes

- Keep the main OCR environment separate from the Laya policy environment.
- Keep sequence-level train/validation splits intact to avoid page/sequence leakage.
- Do not use test ground truth during inference.
- Do not treat unmatched CTD regions as preprocessing positives or negatives without semantic justification.
- Treat Mermaid and other visualizations as debugging outputs, not the canonical representation.
- Do not scale an experiment to the full development set until the small-sample behavior has been inspected.
- Prefer evidence-driven changes over architecture changes based only on intuition.

---

# 18. Next update template

Use this template after each significant experiment:

## Experiment: `<name>`

**Problem:**

**Hypothesis:**

**Change implemented:**

**Dataset / sequence sample:**

**Command / configuration:**

**Observed result:**

**What the result means:**

**What failed / unexpected behavior:**

**Decision:**

**Next experiment:**

**Files changed:**


---

# 19. 2026-09-28 — Proceeding to a broader oracle benchmark

The first corrected oracle run was intentionally kept small: one development sequence only. Detailed inspection showed that all 8 matched examples selected `NONE`, while several difficult regions remained poorly recognized.

This result is **not sufficient evidence to reject preprocessing**, because one sequence may simply contain mostly easy horizontal text. It is also not sufficient evidence to train Laya, because the resulting label distribution would be degenerate (`NONE` only).

Therefore the next experiment is a small multi-sequence benchmark before any model training.

## Hypothesis

Preprocessing may still be useful on other manga sequences. We need to measure whether useful non-`NONE` oracle actions appear across a broader sample.

## Next test

Run the corrected oracle generator on a small, reproducible sample of additional development sequences and aggregate:

- number of matched oracle examples,
- number of unmatched regions,
- oracle-action counts,
- average baseline CER,
- average oracle CER,
- total CER improvement from the selected action.

## Decision rule

- If non-`NONE` actions occur with meaningful frequency and provide real CER gains, preprocessing remains a viable Laya policy task.
- If almost everything remains `NONE`, preprocessing should not be forced into the learned-policy role.
- If high-CER examples are common but transformations rarely help, investigate line segmentation/order and recognition strategy instead.

## Status

No Laya fine-tuning yet. No full 80-sequence oracle generation yet. The next step is evidence collection across a small sequence sample.


# 11. Multi-sequence oracle benchmark — evidence after four sequences

## 11.1 Why this experiment was run

The corrected oracle generator worked on the first sequence, but that sequence produced only `NONE` oracle actions. We therefore tested three additional development sequences before deciding whether preprocessing was a useful enough problem for Laya.

The development sample was:

```text
seq_2032620aa4e4ac7f
seq_952f154fb1505883
seq_7db7a000adeceacf
seq_9032e5551cf1a9d2
```

## 11.2 Results

Across the four sequences:

- matched oracle regions: **91**
- unmatched CTD regions: **25**
- raw-best non-NONE actions: **16 / 91**
- final oracle non-NONE actions under the current `MIN_CER_GAIN = 0.03` rule: **7 / 91**
- final oracle distribution:
  - `NONE`: 84
  - `UPSCALE`: 2
  - `THRESHOLD`: 5
  - `CONTRAST`: 0
  - `ROTATE_CW`: 0
  - `ROTATE_CCW`: 0

The first sequence produced 8 oracle rows and all were `NONE`.

The second sequence produced 30 oracle rows; one selected `UPSCALE`.

The third sequence produced 27 oracle rows and all were `NONE`.

The fourth sequence produced 26 oracle rows; five selected `THRESHOLD` and one selected `UPSCALE`.

## 11.3 Important observation about the gain threshold

The `0.03` minimum CER-improvement threshold is our heuristic, not a property of the task.

Without that threshold, 16 regions had a non-NONE raw best action:

```text
UPSCALE     8
THRESHOLD   6
CONTRAST    2
```

After requiring at least `0.03` absolute CER improvement, only 7 remained:

```text
THRESHOLD   5
UPSCALE     2
```

This means our current oracle label distribution is strongly affected by the threshold we chose.

Therefore, the 7/91 figure must not be interpreted as “only 7 regions benefit from preprocessing.” It is specifically “7 regions showed at least 0.03 absolute CER improvement under the current candidate set.”

## 11.4 What the evidence says

The evidence now shows that preprocessing is **useful but sparse**.

Most matched regions are already handled well by the rectified baseline. Some difficult cases improve with thresholding or upscaling. However, many high-CER cases remain high-CER because of other causes such as line ordering, segmentation, or very difficult/tiny text.

Preprocessing therefore should not be treated as a universal OCR repair mechanism.

## 11.5 Decision

Do **not** train Laya yet.

Do **not** discard preprocessing either.

Before constructing the final Laya training set, test how sensitive the oracle is to the CER-gain threshold and determine whether preprocessing improvements are predictable from features available at inference time.

We should retain both:

- `best_raw_action`
- `oracle_action`

because the distinction between “best candidate” and “best candidate with a meaningful gain threshold” is useful evidence.

## 11.6 Broader architectural lesson

The OCR pipeline is now better understood as multiple distinct subproblems:

```text
CTD localization
       ↓
line segmentation
       ↓
line reading order
       ↓
text recognition
       ↓
story-text / SFX filtering
       ↓
speaker grounding
       ↓
cross-page character identity
```

Preprocessing addresses only one part of this chain.

## 11.7 Next experiment

Run a threshold-sensitivity analysis on the existing oracle results:

```text
gain threshold = 0.00
gain threshold = 0.01
gain threshold = 0.02
gain threshold = 0.03
gain threshold = 0.05
```

Compare:

- number of non-NONE examples,
- action distribution,
- average CER improvement,
- geometry/recognition characteristics of the examples.

Then decide whether:

```text
A) keep preprocessing as a small Laya policy,
B) redesign the policy as OCR-strategy selection, or
C) move the learned decision component to another failure mode such as reading-order selection.
```


# 12. Threshold-sensitivity analysis — deciding whether preprocessing is a strong Laya task

## 12.1 Why this experiment was run

The four-sequence oracle benchmark showed that most regions selected `NONE`, while a smaller number had measurable gains from `UPSCALE`, `CONTRAST`, or `THRESHOLD`. However, the final oracle labels depended on an arbitrary minimum CER-gain threshold of `0.03`.

We therefore evaluated the existing oracle results at gain thresholds of `0.00`, `0.01`, `0.02`, `0.03`, and `0.05`, without rerunning OCR.

## 12.2 Results

There were **91 matched regions** across the four development sequences.

At threshold `0.00`, the raw best-action distribution was:

```text
NONE        75
UPSCALE      8
CONTRAST     2
THRESHOLD    6
ROTATE_CW    0
ROTATE_CCW   0
```

Mean CER changed from approximately **0.0997** for the baseline to **0.0877** when every raw-best preprocessing action was accepted, an average improvement of about **0.0120 CER**.

As the minimum gain threshold increased:

```text
threshold   NONE   UPSCALE   CONTRAST   THRESHOLD   mean CER   mean gain
0.00         75       8          2           6        0.0877      0.0120
0.01         75       8          2           6        0.0877      0.0120
0.02         82       3          1           5        0.0888      0.0109
0.03         84       2          0           5        0.0894      0.0103
0.05         86       0          0           5        0.0903      0.0094
```

## 12.3 Important pattern

The strongest preprocessing improvements are concentrated in a small number of examples, especially `THRESHOLD` cases.

The largest observed gains were approximately:

```text
THRESHOLD   +0.364
THRESHOLD   +0.214
THRESHOLD   +0.125
THRESHOLD   +0.083
THRESHOLD   +0.071
UPSCALE     +0.038
UPSCALE     +0.037
```

At the same time, several examples had alternative preprocessing actions with small gains that disappeared under the `0.03` threshold.

## 12.4 Interpretation

Preprocessing is **not useless**, but it is a sparse intervention rather than the dominant OCR problem in the sampled data.

The mean CER improvement is modest because most regions are already recognized reasonably well and many difficult regions are difficult for reasons that preprocessing does not solve.

The evidence therefore does not support making Laya a simple six-way preprocessing classifier over every text region.

## 12.5 Decision

Do not train Laya yet on the current preprocessing oracle.

Keep the preprocessing experiments and oracle data because they reveal useful edge cases and may support a later retry/gating mechanism.

The next question should be whether we can **predict when preprocessing is likely to help** using information available before retrying OCR, such as:

- baseline OCR confidence,
- region size and aspect ratio,
- line count,
- orientation,
- character-density/visual statistics,
- baseline text quality signals.

This changes the learned task from:

```text
Which preprocessing action should every crop receive?
```

to:

```text
Is a preprocessing retry worth doing, and which retry is appropriate?
```

That is a more realistic decision problem and avoids training a model on a highly imbalanced action distribution without evidence that the policy is predictable.

## 12.6 Next experiment

Build a **preprocessing-helpfulness analysis** from the existing 91 oracle rows.

Group rows by baseline recognition quality and geometric features, then inspect whether large oracle gains correlate with measurable input features.

The outcome will determine whether preprocessing remains a Laya policy or whether Laya should be moved to a more informative decision point such as OCR strategy selection or reading-order resolution.

# 12. Threshold sensitivity analysis — four-sequence oracle evidence

## 12.1 Result

The corrected oracle was evaluated across four development sequences with 91 matched CTD regions.

Raw best preprocessing actions:

```text
NONE        75
UPSCALE      8
CONTRAST     2
THRESHOLD    6
ROTATE_CW    0
ROTATE_CCW   0
```

Using different minimum-CER-gain thresholds produced:

```text
threshold   NONE   UPSCALE   CONTRAST   THRESHOLD
0.00          75       8          2           6
0.01          75       8          2           6
0.02          82       3          1           5
0.03          84       2          0           5
0.05          86       0          0           5
```

At a threshold of 0.00, mean CER changed from 0.0997 to 0.0877, an average improvement of 0.0120. At 0.03, mean CER changed from 0.0997 to 0.0894, an average improvement of 0.0103.

## 12.2 Interpretation

The data shows that preprocessing sometimes produces substantial improvements, but the improvements are sparse.

The strongest observed improvements were thresholding cases. The three largest CER gains were approximately 0.364, 0.214 and 0.125.

At the same time, several difficult OCR regions remained difficult even after trying all current preprocessing candidates. This confirms that some errors originate from line ordering, segmentation, or recognition difficulty rather than preprocessing.

## 12.3 Decision

Do not fine-tune Laya on the current 91 rows.

Do not discard the preprocessing hypothesis yet.

The next question is whether the useful preprocessing cases are predictable from information available at inference time, such as OCR confidence and crop geometry.

The target for the next analysis will be a binary label such as:

```text
HELPFUL = raw-best preprocessing action improves CER by >= 0.02
NOT_HELPFUL = otherwise
```

The analysis must not use `baseline_cer` as an input feature because baseline CER depends on the ground-truth target and is unavailable at test time. It may be used only for defining/evaluating the label.

## 12.4 Next experiment

Run an interpretable feature-predictability analysis using the existing 91 rows. Use sequence-level cross-validation so pages from the same three-page sequence are never split between training and validation.

Candidate inference-time features:

```text
baseline OCR confidence
crop width / height
aspect ratio
area ratio
line count
mean line height
OCR character count
OCR word count
punctuation ratio
orientation
```

The purpose is not to build the final classifier. It is to determine whether a learnable signal exists before committing to a Laya policy.

# 13. Preprocessing predictability analysis — four-sequence diagnostic

## 13.1 Question

After the threshold-sensitivity experiment, the next question was whether the rare cases where preprocessing helps can be predicted from information available before retrying OCR.

We defined an exploratory binary target:

```text
HELPFUL = best tested preprocessing action improves CER by >= 0.02
NOT_HELPFUL = otherwise
```

The analysis used the existing 91 matched regions and did not rerun OCR.

## 13.2 Class distribution

```text
HELPFUL      9
NOT_HELPFUL 82
```

The positive class is therefore sparse: roughly 10% of the current sample.

Among the 9 helpful cases:

```text
UPSCALE      3
CONTRAST     1
THRESHOLD   5
```

No rotation examples were found.

## 13.3 Feature evidence

The helpful cases tended to have lower baseline recognition confidence than the non-helpful cases:

```text
helpful mean confidence      0.9069
not-helpful mean confidence  0.9550
```

They also tended to have fewer OCR characters and words, and larger mean line height:

```text
mean line height: 38.39 vs 20.87
OCR chars:        23.33 vs 49.48
OCR words:         5.00 vs  9.35
```

These are descriptive differences only; the sample is too small for strong conclusions.

## 13.4 Grouped single-feature test

Leave-one-sequence-out evaluation was used so that all regions from one manga sequence were held out together.

Always predicting NOT_HELPFUL gives:

```text
balanced accuracy = 0.500
F1                  = 0.000
```

The strongest individual diagnostic feature was baseline recognition score:

```text
balanced accuracy = 0.711
F1                  = 0.343
recall              = 0.667
```

OCR punctuation ratio was the second strongest tested feature:

```text
balanced accuracy = 0.694
F1                  = 0.292
recall              = 0.778
```

These results show that there is some measurable signal associated with preprocessing usefulness, especially baseline OCR confidence, but this is only a four-sequence diagnostic and is not sufficient evidence for a production policy.

## 13.5 Interpretation

The experiment changes the conclusion from “preprocessing is too sparse to be useful” to a more precise statement:

> Preprocessing improvements are sparse, but they are not completely random with respect to inference-time signals.

The baseline OCR confidence appears to contain useful information about whether a retry may be worthwhile. However, the current feature test is intentionally simple and the positive class is very small.

Importantly, this does **not** yet establish that Laya should be trained. A four-sequence, 91-region diagnostic can demonstrate a hypothesis, but not validate a final learned policy.

## 13.6 Decision

Keep preprocessing as a possible learned/gated component, but do not make it the primary Laya training target yet.

The next experiment should test a small, conventional learned baseline on the same structured features. This gives us a reference point before introducing Laya.

The baseline should answer:

```text
Can a small classifier predict HELPFUL vs NOT_HELPFUL
using only inference-time features and grouped validation?
```

If the conventional model cannot generalize beyond the simple confidence signal, introducing Laya for this job would add complexity without evidence of benefit.

If the conventional model shows useful generalization, Laya can later be tested as the decision model or policy layer against that baseline.

## 13.7 Next experiment

Train/evaluate a small tabular baseline such as logistic regression or a shallow tree ensemble using sequence-level cross-validation.

Compare it against:

```text
always NOT_HELPFUL
single-feature confidence rule
small multi-feature classifier
```

Only after this comparison should we decide whether to fine-tune Laya for preprocessing/retry gating.


# 12. Preprocessing policy predictability — conventional ML baseline

## 12.1 Why this experiment was run

The previous four-sequence oracle analysis showed that preprocessing can improve OCR on a small subset of regions. Before using Laya, we tested whether the usefulness of preprocessing could be predicted from features that would be available at inference time.

## 12.2 Results

The dataset contained:

- 91 matched regions
- 9 regions labeled `HELPFUL` using the exploratory rule: preprocessing gain >= 0.02 CER
- 82 regions labeled `NOT_HELPFUL`

Sequence-level leave-one-sequence-out evaluation was used.

Results:

```text
ALWAYS_NOT_HELPFUL    BA=0.500  F1=0.000
CONFIDENCE_ONLY       BA=0.711  F1=0.343
LOGISTIC_REGRESSION   BA=0.520  F1=0.154
SHALLOW_TREE          BA=0.606  F1=0.273
```

The confidence-only rule therefore generalized better in this small sample than either multi-feature model.

## 12.3 Interpretation

The current evidence suggests that most of the predictable signal is already captured by the OCR recognition confidence.

Adding the current geometry and OCR-text features did not improve grouped validation performance.

This is important because it establishes a conventional baseline that any Laya-based policy would need to improve upon.

However, the experiment is small:

- only four sequences were used;
- only nine positive examples existed;
- the helpful/not-helpful target depends on the current preprocessing candidate set and the chosen 0.02 CER threshold.

Therefore these numbers are diagnostic, not a final conclusion about the full dataset.

## 12.4 Decision

Do not fine-tune Laya for the current preprocessing-helpfulness target yet.

Do not spend the remaining development effort trying to make this particular feature set more complex without first increasing the evidence base.

The current preprocessing subsystem should remain available as a deterministic retry mechanism, with OCR confidence as a baseline trigger.

The learned decision component should now be investigated on a problem closer to the actual task objective.

## 12.5 Architectural direction

A more central decision problem is text-region/story-text interpretation:

```text
CTD region
   ↓
OCR
   ↓
structured evidence
   ├─ recognized text
   ├─ OCR confidence
   ├─ geometry
   ├─ line count
   └─ page context
   ↓
learned decision
   ↓
story text / SFX / other excluded text
```

This maps more directly to the task requirement to include dialogue, thoughts, narration, vocalisations, and other story text while excluding SFX and non-story text.

The main caution is label quality: CTD detections that do not match a ground-truth utterance are not automatically valid negative examples. Some are OCR failures or segmentation failures. Future data construction must therefore distinguish reliable hard negatives from unmatched/uncertain regions.

## 12.6 Next experiment

Investigate a sequence-aware **story-text classification** dataset.

First prototype the labeling/alignment strategy on a small sample:

1. map GT utterances to detected CTD regions;
2. identify reliable negative regions from obvious non-story text;
3. keep uncertain/unmatched regions separately;
4. test whether simple structured features can separate story text from reliable negatives;
5. only then consider Laya fine-tuning.

This keeps the learned component directly connected to the final manga extraction objective instead of forcing Laya into a weak preprocessing policy.


# 13. Research review — where the next engineering effort should go

## 13.1 External research reviewed

The next-step design was checked against recent and established manga/comics research.

### Manga text-bubble reading order

Kovanen and Aizawa proposed a layered method for determining manga text-bubble reading order using text-bubble position and image information. Their reported evaluation used 1,769 manga pages and 14,726 manually annotated text positions, with over 95% transition accuracy. This supports treating reading order as a distinct layout problem rather than assuming detector order or a simple global sort is sufficient.

### Comic speaker-to-text association

Manga109Dialog provides 132,692 speaker-to-text pairs and treats speaker detection as an explicit relation-prediction problem. Its proposed scene-graph-based method also uses frame information and reading order. This supports keeping speaker grounding as a sequence-level relation problem rather than reducing it to nearest-character distance.

### Annotation quality / dialogue vs onomatopoeia

Manga109-v2026 reports that existing manga dialogue annotations contain transcription errors, missing text regions, overlap between dialogue and onomatopoeia, and under-segmented speech balloons. This closely matches problems already observed in the current project and reinforces the need to keep uncertain detections separate from reliable positives/negatives.

### Speech-bubble detection

An available Apache-2.0 model, `ogkalu/comic-speech-bubble-detector-yolov8m`, is documented as a YOLOv8-medium detector trained on about 8,000 manga, webtoon, manhua, and Western comic images at 1024px input size. It is a plausible additional feature source, but its performance must be validated on this dataset before integrating it.

### Laya

Current Laya documentation explicitly positions the model as a typed-decision engine and states that its base checkpoints should be specialised for the target domain through fine-tuning. The project now has the fine-tuning notebook and reports substantially better domain-specific results after fine-tuning. Therefore Laya should be used only after we have a well-defined, trustworthy decision dataset.

## 13.2 Research-driven architectural conclusion

The current evidence suggests that the highest-value immediate problem is not image preprocessing.

The project should first improve the structural pipeline:

```text
CTD localization
       ↓
speech/text region representation
       ↓
reading order
       ↓
story-text filtering
       ↓
speaker grounding
       ↓
cross-page identity
       ↓
JSONL output
```

The preprocessing experiment remains useful as a deterministic retry path, but it should not drive the main learned architecture.

## 13.3 Immediate experiment selected

We will benchmark reading-order heuristics on the existing matched data before implementing a learned ordering model.

The benchmark compares:

```text
DETECTOR
TOP_LEFT
TOP_RIGHT
ROW_LTR
ROW_RTL
```

Metrics:

- pairwise ordering accuracy
- exact page ordering accuracy
- transition-style accuracy

This experiment requires no new model download and directly tests one of the failure modes already observed in OCR output.

## 13.4 Planned next stages after reading-order benchmark

If simple spatial ordering improves materially:

1. implement a deterministic reading-order layer;
2. add line-level ordering inside CTD blocks;
3. validate against all development sequences.

Then:

4. validate a speech-bubble detector and use bubble overlap/type as a feature;
5. construct reliable story-text positives and reliable non-story/SFX negatives;
6. benchmark conventional classifiers;
7. define a Laya decision schema only after the label quality is adequate;
8. fine-tune Laya and compare it against the conventional baseline;
9. move to speaker grounding and cross-page character identity.

## 13.5 Decision discipline

No component is added merely because an external paper or model exists.

Every major component must pass:

```text
hypothesis
→ small benchmark
→ measured result
→ comparison against baseline
→ documented decision
→ only then scale up
```

This is now the project's standard progression.

---

# 15. Architecture lock — CTD + OCR + Qwen3-VL + Laya decision layer

**Date:** 2026-09-28

## 15.1 Final architecture decision

The project architecture is now locked as follows:

```text
3-page sequence
      ↓
LangGraph control plane
      ↓
parallel page perception
      ↓
CTD localization
      ↓
PaddleOCR recognition
      ↓
selective OCR retries / alternate preprocessing
      ↓
Qwen3-VL visual transcription + verification + semantic/spatial evidence
      ↓
Candidate Bank
      ↓
Laya learned decision layer
      ↓
story-text filtering + reading order + speaker grounding
      ↓
character identity graph / cross-page consistency
      ↓
validator
      ↓
JSONL
```

The important structural correction is that OCR and Qwen3-VL are **parallel evidence sources** for the decision layer, not a simple sequential `OCR → VLM → final text` chain.

## 15.2 Role of Laya

Laya is not being used as the primary OCR model and is not expected to infer an answer directly from raw pixels.

Its intended role is to select among structured candidate outputs using evidence such as:

- OCR candidate text,
- OCR recognition confidence,
- alternate OCR candidate text,
- VLM transcription,
- VLM visual evidence,
- geometry/crop features,
- text-type evidence,
- surrounding context.

A conceptual decision is:

```text
OCR_BASE
OCR_UPSCALE
OCR_THRESHOLD
VLM
FUSED
    ↓
  Laya
    ↓
selected candidate + decision probability
```

The probability returned by Laya is not assumed to be calibrated automatically. Calibration will be evaluated on held-out development data.

## 15.3 Qwen3-VL smoke test

Qwen3-VL-4B-Instruct was successfully installed and loaded locally using 4-bit quantization.

Observed behavior:

- model weights downloaded successfully,
- local CUDA inference succeeded,
- individual manga crops were processed successfully,
- the model returned structured JSON in the smoke test,
- GPU allocated memory during the successful crop test was approximately 2.72 GiB.

For the crop containing `HUH? DOES`, PaddleOCR and Qwen3-VL agreed on the transcription. Qwen reported high visual confidence, but this self-reported number is treated only as evidence, not as a calibrated correctness probability.

## 15.4 Whole-page VLM test — failure mode discovered

A first VLM test used the entire manga page.

Qwen produced a long transcription containing text from multiple manga regions in one output. This is unsuitable for candidate-level OCR resolution because it breaks the mapping:

```text
CTD block → OCR result → VLM result → GT
```

**Decision:** the final VLM OCR-verification experiment must operate at the **localized CTD block/utterance level**, not as whole-page free-form transcription.

## 15.5 Initial line-level OCR/VLM benchmark — interpretation changed

The first benchmark sent individual CTD line crops to Qwen3-VL.

Across 20 tested lines, OCR/VLM exact agreement was 15%.

This was **not interpreted as a direct quality comparison**, because the experimental units were mismatched:

- CTD/PaddleOCR produced individual line outputs,
- Qwen often reconstructed a larger phrase using visual context,
- the task ground truth is organized around story utterances/text regions rather than isolated OCR fragments.

Examples demonstrated this clearly. PaddleOCR produced fragments such as `HE'S`, `DEOING`, `IT Looks ALY`, while Qwen reconstructed larger phrases such as `HE'S DOING REALLY`.

**Decision:** do not train or evaluate Laya from the line-level agreement number.

## 15.6 Corrected experiment: block-level evaluation

The next benchmark is explicitly block-level and page-traceable.

For every tested region it records:

```text
sequence_id
page_index
page_path
block_id
bbox
block_crop_path
OCR text
OCR confidence
VLM text
VLM confidence
GT text
OCR CER
VLM CER
better candidate
```

The experiment also saves the exact block crop and individual line crops under a sequence/page/block directory tree.

This allows every comparison to be traced back to the exact page and CTD region that produced it.

## 15.7 Why block-level evaluation is now preferred

The development oracle examples show that a complete story utterance can span several CTD lines. Some observed OCR errors are therefore caused by line segmentation, line grouping, or ordering rather than character recognition alone.

For example, a target such as:

```text
HE'S DOING REALLY WELL. IT LOOKS LIKE HE'S BEEN WORKING HARD FOR THE FALL TOURNAMENT.
```

can be fragmented across many OCR lines. This means that a line-perfect comparison can be misleading when the real target is the complete utterance.

**Decision:** benchmark and eventually train candidate selection at the smallest unit that aligns with the target story utterance/block.

## 15.8 Current next step

Run the new block-level benchmark on a small number of development pages before any Laya training.

The immediate question is:

> Does Qwen3-VL provide complementary evidence to PaddleOCR at the utterance/block level, and in which failure cases?

Only after this is quantified should the candidate-selection dataset for Laya be constructed.

---

# 16. Current project state after architecture lock

```text
Data/schema layer                ✅
Sequence loader                  ✅
CTD localization                 ✅
PaddleOCR                        ✅
OCR preprocessing candidates     ✅
Corrected preprocessing oracle   ✅
Laya local checkpoint            ✅
Qwen3-VL local inference         ✅
Whole-page VLM test              ✅ diagnostic only
Line-level VLM test              ✅ diagnostic only
Block-level VLM benchmark        ⏳ next
Candidate Bank                   ⏳ next
Laya candidate-selection dataset ⏸️ after benchmark
Laya fine-tuning                 ⏸️
Story-text resolver              ⏸️
Reading-order resolver            ⏸️
Speaker grounding                ⏸️
Character identity graph         ⏸️
Full LangGraph pipeline          ⏸️
Final test inference             ⏸️
```


---

# 17. Block-level VLM benchmark — first successful execution and debugging

## 17.1 Execution after CTD geometry fix

The first replacement benchmark failed because the script accessed:

```python
block.bounding_box
```

but this CTD `TextBlock` implementation exposes:

```python
block.bounding_rect()
```

The error was:

```text
AttributeError: 'TextBlock' object has no attribute 'bounding_box'.
Did you mean: 'bounding_rect'?
```

The helper was therefore changed to call `bounding_rect()` when it is callable and convert its `[x, y, width, height]` representation into `[x1, y1, x2, y2]`.

**Result:** CTD block processing proceeded successfully on the next run.

## 17.2 OCR sanity check passed

The corrected benchmark performed an explicit PaddleOCR sanity check before running the full block benchmark.

Observed:

```text
PyTorch           : 2.5.1+cu121
ONNX Runtime      : 1.26.0
ORT providers     : TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
CUDA available    : True
CUDA device       : NVIDIA GeForce RTX 4050 Laptop GPU

Sanity OCR text : 'HUH? DOES'
Sanity OCR score: 0.9922
PaddleOCR sanity check: PASS
```

This establishes that the current OCR environment is functioning and that the previous all-empty OCR output was caused by benchmark implementation issues rather than a general PaddleOCR failure.

## 17.3 Block-level OCR is now producing real candidates

After the geometry correction, the benchmark produced non-empty OCR candidates on many localized CTD blocks.

Examples from the run include:

```text
OCR: 'HUH? DOES K KOMBA SEEM DI DIFFFERENT THANUSAL?'
OCR: 'OUT F BLOOD.'
OCR: 'RIGHT HER ERE! THISI IS I WHHERE IT WAPENS!'
OCR: 'OKAWA- SE NPAI Y oOu LOOK So coOL!'
OCR: "IT'S ALL FOR THE SAKE OF OUR CLB. WE'RE GOING TO ANALYZE IT IN DETAIL LATER."
```

These outputs demonstrate that the block pipeline is correctly aggregating CTD line recognition into a block-level OCR candidate, although OCR still contains substantial recognition/spacing errors on difficult crops.

The benchmark also correctly recognized obvious excluded/non-story text in at least one observed case:

```text
OCR      : 'SFX: HOOFBEATS'
VLM TYPE : sound_effect
GT       : None
```

This is useful evidence for later story-text filtering, but it is not yet a benchmark metric.

## 17.4 Qwen3-VL block inference is functioning

Qwen3-VL produced localized block-level transcriptions and text-type classifications.

Examples include:

```text
'HUH? DOES KOMABA SEEM DIFFERENT THAN USUAL?'
"HE'S DOING REALLY WELL. IT LOOKS LIKE HE'S BEEN WORKING HARD FOR THE FALL TOURNAMENT."
'OUT FOR BLOOD.'
'RIGHT HERE! THIS IS WHERE IT HAPPENS!'
'OKAWA-SENPAI, YOU LOOK SO COOL!'
"IT'S ALL FOR THE SAKE OF OUR CLUB. WE'RE GOING TO ANALYZE IT IN DETAIL LATER."
'SO CLOSE!!'
```

The model was able to distinguish a visible SFX crop as:

```text
text_type = sound_effect
```

and dialogue crops as:

```text
text_type = dialogue
```

This confirms that the localized block-level VLM interface is operational.

## 17.5 Benchmark metrics are currently invalid because GT correspondence is broken

Although the CTD, OCR, and VLM stages executed, the benchmark reported:

```text
Oracle rows: 8
Loaded oracle rows: 2

Reliable matches: 0
Mean OCR CER  : N/A
Mean VLM CER  : N/A
```

This is a critical implementation inconsistency.

The existing oracle builder itself reports 8 oracle rows for this sequence, but the block benchmark's `load_oracle()` function loaded only 2 rows.

Therefore the benchmark could not attach ground-truth utterances to the processed CTD blocks. Every block was consequently printed as:

```text
GT : None
GT : UNMATCHED
```

The resulting `0` reliable matches does **not** indicate that OCR or Qwen failed. It indicates that the benchmark's oracle-loading/correspondence layer is incorrect.

## 17.6 Consequence for Laya development

The current run must not be used as training data or as evidence for choosing OCR versus VLM.

Specifically, the following must be discarded as benchmark metrics:

```text
Reliable matches = 0
OCR better = 0
VLM better = 0
Mean OCR CER = N/A
Mean VLM CER = N/A
```

No Laya candidate-selection dataset should be constructed from this run.

The candidate-generation pipeline itself can continue to be developed, because CTD localization, block cropping, PaddleOCR recognition, and localized Qwen3-VL inference are now executing.

## 17.7 Debugging priority changed

The next technical priority is no longer OCR execution.

The order is now:

```text
1. Inspect actual preprocessing-oracle JSONL schema
2. Fix load_oracle() parsing/keying
3. Verify:
       oracle builder row count == loader row count
4. Verify each block receives the intended GT row
5. Re-run block benchmark on the same sequence
6. Only then compute OCR CER / VLM CER
7. Only then construct Candidate Bank training examples
```

The benchmark should include an explicit integrity assertion such as:

```text
expected oracle rows from builder
        ==
rows successfully loaded
```

and should stop before model comparison when this condition is false.

## 17.8 Current project state after this run

```text
Data/schema layer                ✅
Sequence loader                  ✅
CTD localization                 ✅
CTD block geometry handling      ✅ fixed
PaddleOCR line recognition       ✅
OCR sanity gate                  ✅
Block-level OCR aggregation      ✅
OCR preprocessing candidates     ✅
Corrected preprocessing oracle   ✅
Laya local checkpoint            ✅
Qwen3-VL local inference         ✅
Whole-page VLM test              ✅ diagnostic only
Line-level VLM test              ✅ diagnostic only
Block-level VLM inference        ✅
Block-level benchmark metrics    ❌ blocked by oracle loader
Candidate Bank                   ⏳
Laya candidate-selection dataset ⏸️
Laya fine-tuning                 ⏸️
Story-text resolver              ⏸️
Reading-order resolver            ⏸️
Speaker grounding                ⏸️
Character identity graph         ⏸️
Full LangGraph pipeline          ⏸️
Final test inference             ⏸️
```

## 17.9 Immediate next experiment

Do **not** expand the benchmark to all 80 development sequences yet.

First make the oracle mapping deterministic and verifiable on:

```text
seq_2032620aa4e4ac7f
```

Then rerun the same 3-page block benchmark.

Success criteria for the next run are:

```text
oracle builder rows == oracle loader rows
GT is non-null for every matched oracle block
CTD block -> GT correspondence is stable
OCR text and VLM text are compared against the same GT
CER values are finite and reproducible
```

Only after these checks pass should the benchmark be expanded to additional sequences.


# 18. Oracle loader fix and second block benchmark run

## 18.1 Oracle loader count is now fixed

The `load_oracle()` change from `block_id` fallback logic to the actual oracle schema field `block_index` successfully removed the dictionary-key collision.

The benchmark now reports:

```text
Oracle rows: 8
Loaded oracle rows: 8
```

The oracle builder also continues to report the expected 8 matched rows and 6 unmatched CTD blocks for this sequence.

This confirms that the previous `8 -> 2` loader-collapse bug has been fixed.

## 18.2 OCR and Qwen runtime remain operational

The same run passed the PaddleOCR sanity gate:

```text
Sanity OCR text : 'HUH? DOES'
Sanity OCR score: 0.9922
PaddleOCR sanity check: PASS
```

The runtime environment continued to report CUDA availability and the RTX 4050 Laptop GPU, and Qwen3-VL loaded successfully.

## 18.3 The remaining problem is oracle-to-runtime block correspondence

The new run did **not** fully validate the benchmark yet.

The benchmark processed 13 CTD blocks, but only 3 received an oracle row. The three rows that did receive GT were attached to the wrong runtime page positions:

```text
Runtime page 2 / block 0
GT: RIGHT HERE! THIS IS WHERE IT HAPPENS!

Runtime page 2 / block 1
GT: OKAWA-SENPAI, YOU LOOK SO COOL!

Runtime page 2 / block 2
GT: IT'S ALL FOR THE SAKE OF OUR CLUB...
```

Those targets belong to the third image/page in the sequence, not the second image/page. Meanwhile, the expected page-1 and page-3 runtime blocks remained `GT=None`.

Therefore the benchmark currently demonstrates a **page-index / block-key convention mismatch between the oracle and the benchmark runtime**. The exact implementation source of that offset must be fixed before quality metrics are considered valid.

## 18.4 Current numerical output is invalid as a model-quality result

The run reported:

```text
Blocks processed  : 13
Reliable matches  : 3
OCR better        : 1
VLM better        : 1
Equal             : 1
Mean OCR CER      : 0.8943
Mean VLM CER      : 1.4162
```

These numbers must **not** be interpreted as OCR-vs-VLM quality measurements, because the three attached GT rows are cross-page assignments rather than verified block-to-GT correspondences.

In particular, the apparent `VLM better` / `OCR better` counts are not training evidence.

## 18.5 Important implementation lesson

Matching only on `(page_index, block_index)` is safe only when the page-index convention is guaranteed to be identical across:

```text
CTD runtime
benchmark loop
oracle builder
saved results
```

The benchmark must explicitly verify this invariant.

For every oracle row, the validation should compare at least:

```text
sequence_id
page identity / image path
page index
block index
bbox
```

The expected mapping for `seq_2032620aa4e4ac7f` is:

```text
01.png / block 0 -> HUH? DOES KOMABA SEEM DIFFERENT THAN USUAL?
01.png / block 1 -> HE'S DOING REALLY WELL...
01.png / block 4 -> OUT FOR BLOOD.

03.png / block 0 -> RIGHT HERE! THIS IS WHERE IT HAPPENS!
03.png / block 1 -> OKAWA-SENPAI, YOU LOOK SO COOL!
03.png / block 2 -> IT'S ALL FOR THE SAKE OF OUR CLUB...
03.png / block 3 -> DO YOU HAVE TO KEEP WATCHING THAT? IT'S EMBARRASSING.
03.png / block 4 -> SO CLOSE!!
```

## 18.6 Current debugging state

The debugging sequence is now:

```text
Oracle JSON schema             ✅
Oracle row loading             ✅ 8/8
CTD localization               ✅
PaddleOCR runtime              ✅
Qwen3-VL runtime               ✅
Oracle page/block correspondence ❌
Valid OCR-vs-VLM benchmark     ⏸️
Candidate Bank dataset         ⏸️
Laya training                 ⏸️
```

The next implementation task is therefore **not** another model test. It is to make the oracle lookup use the exact same page identity and block identity as the CTD runtime.

## 18.7 Next experiment

Fix the page-index convention and add a hard correspondence assertion before CER calculation.

A successful run must show:

```text
Oracle rows: 8
Loaded oracle rows: 8

01.png/block 0 -> correct GT
01.png/block 1 -> correct GT
01.png/block 4 -> correct GT

03.png/block 0 -> correct GT
03.png/block 1 -> correct GT
03.png/block 2 -> correct GT
03.png/block 3 -> correct GT
03.png/block 4 -> correct GT
```

Only after that should the benchmark calculate aggregate OCR/VLM metrics or produce Laya labels.

# 19. Corrected block benchmark — correspondence validated

## 19.1 Oracle/page/block correspondence is now validated on the target sequence

The corrected benchmark run for `seq_2032620aa4e4ac7f` loaded all 8 oracle rows and attached the expected ground-truth utterance to the corresponding runtime blocks after applying the 1-based runtime page to 0-based oracle page conversion.

The verified story-text mappings are:

```text
01.png / block 0 -> HUH? DOES KOMABA SEEM DIFFERENT THAN USUAL?
01.png / block 1 -> HE'S DOING REALLY WELL. IT LOOKS LIKE HE'S BEEN WORKING HARD FOR THE FALL TOURNAMENT.
01.png / block 4 -> OUT FOR BLOOD.

03.png / block 0 -> RIGHT HERE! THIS IS WHERE IT HAPPENS!
03.png / block 1 -> OKAWA-SENPAI, YOU LOOK SO COOL!
03.png / block 2 -> IT'S ALL FOR THE SAKE OF OUR CLUB. WE'RE GOING TO ANALYZE IT IN DETAIL LATER.
03.png / block 3 -> DO YOU HAVE TO KEEP WATCHING THAT? IT'S EMBARRASSING.
03.png / block 4 -> SO CLOSE!!
```

The benchmark output now shows non-null GT on these corresponding page/block records rather than the previous cross-page assignments. The output records also preserve the oracle row with the same bbox and block index.

## 19.2 First valid OCR-vs-VLM block benchmark result

The run processed 13 CTD blocks under `--limit-per-page 5`:

```text
Page 1 : 5 blocks
Page 2 : 3 blocks
Page 3 : 5 blocks
Total  : 13 blocks
```

Eight of these blocks correspond to oracle story utterances. The other five blocks have no matching story-text GT in this oracle and therefore are excluded from the CER aggregate.

From the eight verified matched blocks, the per-block metrics are:

```text
Block        OCR CER       VLM CER
01/0         0.2093        0.0000
01/1         1.0000        0.0000
01/4         0.1429        0.0000
03/0         0.2432        0.0000
03/1         0.1613        0.0000
03/2         0.0130        0.0000
03/3         0.1887        0.0377
03/4         0.9000        0.0000
```

Arithmetic mean across these eight matched blocks:

```text
Mean OCR CER : 0.3573
Mean VLM CER : 0.0047
```

On this single sequence, VLM CER is lower than OCR CER on all 8 verified matched blocks.

This is the first benchmark result in the project that is suitable for quantitative model analysis. It is a single-sequence diagnostic, not a global model-quality conclusion.

## 19.3 Observed complementary failure cases

The benchmark confirms several useful candidate-generation patterns.

### OCR failure with successful VLM reconstruction

For `01.png / block 1`, PaddleOCR produced an effectively empty block transcription while Qwen3-VL reconstructed the complete utterance:

```text
GT  : HE'S DOING REALLY WELL. IT LOOKS LIKE HE'S BEEN WORKING HARD FOR THE FALL TOURNAMENT.
OCR : ""
VLM : HE'S DOING REALLY WELL. IT LOOKS LIKE HE'S BEEN WORKING HARD FOR THE FALL TOURNAMENT.
```

### Difficult small text

For `03.png / block 4`:

```text
GT  : SO CLOSE!!
OCR : S
VLM : SO CLOSE!!
```

This is a useful example of visual reconstruction complementing line-level OCR.

### VLM is not universally exact

For `03.png / block 3`, VLM produced a near-match with a hyphenated line break:

```text
VLM : DO YOU HAVE TO KEEP WATCHING THAT? IT'S EMBAR- RASSING.
GT  : DO YOU HAVE TO KEEP WATCHING THAT? IT'S EMBARRASSING.
```

Therefore VLM output still requires normalization/selection logic and should not be treated as an unquestioned oracle.

## 19.4 Non-GT regions expose the story-text filtering problem

The five unmatched CTD blocks are not all OCR failures. On page 2, for example, Qwen3-VL produced:

```text
SFX: HOOFBEATS          -> text_type = sound_effect
LAST TIME               -> text_type = narration
SUMMER VACATION...      -> text_type = narration
```

while the oracle has zero story utterances for that page.

This demonstrates an important distinction:

```text
visible text
      !=
story text required by the benchmark
```

Therefore future candidate processing needs an explicit story-text eligibility/filtering decision in addition to transcription selection.

## 19.5 Current benchmark status

```text
Oracle schema                 ✅
Oracle loading                ✅ 8/8
Page identity                 ✅
Block identity                ✅
Block-to-GT correspondence    ✅ for target sequence
PaddleOCR runtime             ✅
Qwen3-VL runtime              ✅
First valid OCR/VLM CER       ✅
Candidate Bank                ⏳
Laya training                 ⏸️
```

## 19.6 Next experiment

Do not train Laya from this sequence alone.

Run the same corrected benchmark on:

```text
seq_952f154fb1505883
seq_7db7a000adeceacf
seq_9032e5551cf1a9d2
```

Collect per-block records for:

```text
OCR_NONE
OCR_UPSCALE
OCR_CONTRAST
OCR_THRESHOLD
VLM
```

with geometry/features, OCR recognition scores, VLM evidence, text type, GT correspondence, and CER.

Then aggregate across complete sequences rather than individual pages or random blocks.

## 19.7 Gate before Laya dataset construction

Proceed to Candidate Bank/Laya dataset construction only when the multi-sequence benchmark can answer:

```text
How often is OCR empty or badly corrupted?
How often does preprocessing improve OCR?
How often does VLM provide a materially better candidate?
How often do OCR and VLM disagree?
How often does VLM generate text that is visible but excluded from story text?
```

The intended learning problem is then:

```text
structured candidate/evidence set
              ↓
             Laya
              ↓
selected transcription + eligibility decision
```

rather than a simple fixed rule such as "always choose VLM".

# 20. Expanded block-level OCR vs Qwen3-VL diagnostic — four development sequences

## 20.1 Evidence used

The latest set of four result JSONL files covers these development sequences:

```text
seq_7db7a000adeceacf
seq_952f154fb1505883
seq_9032e5551cf1a9d2
seq_2032620aa4e4ac7f
```

The benchmark processed a limited number of CTD blocks per page rather than exhaustively enumerating every detector region. Therefore the unmatched-row count below must **not** be interpreted as story-text recall or missed-GT rate for the whole page.

## 20.2 Aggregate observed results

Across the four result files:

```text
Processed CTD blocks       : 58
Rows with attached GT      : 47
Unmatched processed rows   : 11

Mean OCR CER (47 matched)  : 0.2508
Mean VLM CER (47 matched)  : 0.0081
```

Among the 47 matched rows:

```text
VLM lower CER : 45
OCR lower CER : 0
Equal         : 2
```

The comparison is still a **diagnostic**, not a final system-model verdict, because it is based on a limited CTD block sample and because the VLM is evaluated only on blocks localized by CTD in this experiment.

## 20.3 Per-sequence results

| Sequence | Processed | GT matched | Unmatched | Mean OCR CER | Mean VLM CER |
|---|---:|---:|---:|---:|---:|
| `seq_7db7a000adeceacf` | 15 | 13 | 2 | 0.2489 | 0.0000 |
| `seq_952f154fb1505883` | 15 | 14 | 1 | 0.2115 | 0.0194 |
| `seq_9032e5551cf1a9d2` | 15 | 12 | 3 | 0.2277 | 0.0060 |
| `seq_2032620aa4e4ac7f` | 13 | 8 | 5 | 0.3573 | 0.0047 |

The arithmetic means above are over matched block rows within each sequence. The aggregate means are arithmetic means over the 47 matched block rows, not averages of sequence means.

## 20.4 What the four-sequence evidence changes

The earlier single-sequence result showed strong Qwen3-VL complementarity. The expanded diagnostic repeats that pattern across the four sampled sequences: on the matched CTD blocks, Qwen3-VL generally reconstructs the utterance more accurately than the current PaddleOCR output.

This still does **not** justify replacing PaddleOCR or the Candidate Bank. It establishes that the recognition branch should preserve multiple evidence sources and that VLM-derived reconstruction is valuable enough to benchmark further.

The result also reinforces the distinction:

```text
recognition quality
        !=
story-text eligibility
```

Several unmatched CTD regions received plausible Qwen3-VL transcriptions even though they were not development-oracle story utterances. Examples included narration and sound effects. The system therefore still needs an explicit eligibility decision.

## 20.5 Four-sequence candidate-bank implication

The current evidence supports retaining the following structure:

```text
CTD localized region
       ├── PaddleOCR candidates
       └── Qwen3-VL evidence
                ↓
          Candidate Bank
                ↓
               Laya
```

Do not collapse the bank to a VLM-only path based on CER alone.

## 20.6 Benchmark integrity issue exposed by reruns

A later rerun of the benchmark hit:

```text
RuntimeError: Oracle/runtime block mismatch: page=1, block=0, IoU=0.836
```

The error indicates that the current hard correspondence assertion is brittle to small CTD bounding-box drift. An IoU of `0.836` is close to the existing `0.85` guard and does not by itself demonstrate a wrong block identity.

The benchmark should therefore distinguish:

```text
block identity verification
bbox drift diagnostics
```

The CTD deduplication threshold must remain conceptually separate from the oracle/runtime verification threshold. The rerun guard should report low-IoU drift and only fail when geometric evidence is insufficient to establish correspondence.

This issue must be resolved before treating repeated multi-sequence benchmark runs as a stable automated regression test.

# 21. Nemotron OCR v2 — standalone capability experiment

## 21.1 Experiment goal

The next OCR experiment intentionally removes CTD and PaddleOCR from the path. The purpose is to measure what Nemotron OCR v2 can do on a complete manga page by itself.

```text
01.png ──┐
02.png ──┼──> Nemotron OCR v2 ──> page regions + text + confidence
03.png ──┘
```

No CTD localization, preprocessing, PaddleOCR, Qwen3-VL, or Laya is involved in this first test.

## 21.2 Why full-page inference is appropriate

Nemotron OCR v2 is an end-to-end OCR system with detector, recognizer, and relational components. The official interface supports merged outputs at word, sentence, and paragraph levels, so a full-page experiment can test both text detection and the model's grouping/layout behavior rather than only recognition on pre-localized crops. citeturn756805search0turn756805search8

For the English manga task, the first run should use the English model variant:

```python
NemotronOCRV2(lang="en")
```

and start with:

```python
merge_level="sentence"
```

The benchmark can then repeat the same pages at `word` level if sentence grouping appears to merge or split manga utterances badly.

## 21.3 Output required from the standalone run

For each page, preserve:

```text
bbox
text
confidence
region_count
runtime_ms
page_index
```

Do not attach CTD block IDs at this stage. The first artifact should expose Nemotron's own regions exactly as returned by the model.

## 21.4 Evaluation after raw inference

Once raw output is obtained:

```text
Nemotron regions
       ↓
geometric GT matching
       ↓
matched / unmatched regions
       ↓
CER + detection coverage diagnostics
       ↓
visual inspection of grouping/order
```

Geometric matching should use IoU/containment against GT boxes rather than relying on Nemotron region index equal to oracle `block_index`.

## 21.5 Environment note

The official Nemotron OCR v2 quickstart documents a Python 3.12 environment and a CUDA/PyTorch-compatible build, with the package compiling a C++ extension. The same documentation also provides a Docker path. citeturn756805search0turn756805search8

Because the project is developed on Windows with a 6 GB RTX 4050 Laptop GPU, the first installation should be done in the project's Linux/WSL or Docker GPU environment rather than modifying the main Windows environment blindly.

## 21.6 Current experiment gate

```text
Four-sequence CTD+PaddleOCR+Qwen diagnostic   ✅ recorded
Nemotron standalone full-page run             ⏳ next
Nemotron vs GT geometric evaluation            ⏳
Nemotron vs CTD localization comparison        ⏳
Nemotron inclusion in Candidate Bank            ⏸️ until measured
Laya training                                  ⏸️
```

# 22. Nemotron OCR v2 — standalone page-01 capability probe

## 22.1 Environment and installation status

The standalone Nemotron environment is now operational in WSL on the D: drive:

```text
Python             3.12.14
PyTorch            2.11.0+cu128
torchvision        0.26.0+cu128
CUDA Toolkit       12.8.61
nvcc               working
CUDA execution     working on RTX 4050 Laptop GPU
Nemotron import    OK
```

The English checkpoint is present locally under `v2_english/` with detector, recognizer, relational weights and charset. The separate RegNet backbone was moved from the WSL home cache to `D:\STORY-AI\.torch\hub\checkpoints` and `TORCH_HOME` was corrected to the D: location.

## 22.2 Full-page probe without CTD

The first complete manga-page test used:

```text
seq_2032620aa4e4ac7f/01.png
```

No CTD, PaddleOCR, preprocessing, Qwen3-VL, or Laya was used.

### Word mode with relational stage

Nemotron returned:

```text
29 text regions
```

The output recovered all three visible dialogue areas and the `OUT FOR BLOOD.` balloon, while also detecting non-story text such as `SNAP` and the large vertical page text.

### Sentence/paragraph comparison

Running the same page at `merge_level="sentence"` and `merge_level="paragraph"` both produced 29 output regions with the same observed fragmentation of the central multi-line dialogue balloon. The central utterance remained split into many regions rather than becoming one useful manga utterance/block.

### Skip-relational word mode

Using the official `skip_relational=True` mode produced:

```text
50 text regions
```

This exposed the raw detector/recognizer behavior more directly. The story balloons were detected as separate word/line regions. The model also detected substantial non-story text, including `SNAP`, the large vertical page lettering, the page number-like `1`, and other surrounding text.

One clear recognition error in a GT story balloon was:

```text
GT      : HUH? DOES KOMABA SEEM DIFFERENT THAN USUAL?
Nemotron: HUH? / DOES / KOMABA / SEEM / DIFFERENT / THAN / UISUL?
```

The model therefore demonstrated independent full-page localization and useful recognition, but its raw word regions are not themselves the final utterance representation required by the benchmark.

## 22.3 Decision from the probe

For the first quantitative standalone benchmark, use Nemotron's raw word-level path:

```text
full page
   ↓
Nemotron detector + recognizer
   ↓
skip_relational=True
   ↓
word regions
   ↓
geometric comparison to GT story blocks
```

This avoids conflating Nemotron's relational grouping behavior with its underlying detection/recognition capability.

The benchmark must report two distinct quantities:

```text
detection coverage:
    how many GT story blocks receive at least one Nemotron region

GT-conditioned recognition:
    CER after collecting Nemotron regions inside the known GT story bbox
```

The second quantity is explicitly GT-conditioned and is not end-to-end story extraction.

## 22.4 Quantitative benchmark implementation

A dedicated script was created:

```text
experiments/ocr/nemotron_fullpage_gt_benchmark.py
```

It runs the three pages of a complete development sequence directly through Nemotron, preserves all raw regions, loads the existing preprocessing oracle using its actual persisted schema (`page_index`, `block_index`, `features.x1..y2`, `target_text`), converts Nemotron normalized boxes to pixels, and performs geometric GT conditioning without CTD.

The script also records:

```text
region_count
runtime_ms
GT block coverage
GT-conditioned CER
GT-conditioned exact match rate
regions inside story GT boxes
regions outside story GT boxes
```

It does not modify the main CTD/PaddleOCR/Qwen pipeline.

## 22.5 Current gate

```text
Qwen/CTD diagnostic baseline                 ✅
Nemotron installation                        ✅
Nemotron sample-page probe                   ✅
Nemotron standalone 3-page quantitative run  ⏳ NEXT
Nemotron vs CTD comparison                   ⏳
Nemotron Candidate Bank inclusion            ⏸️ until measured
Laya training                                ⏸️
```

# 23. Nemotron Standalone 3-Page Quantitative Benchmark — COMPLETED

Sequence:
`seq_2032620aa4e4ac7f`

Model:
`nvidia/nemotron-ocr-v2`

Configuration:
- English checkpoint: `v2_english`
- `merge_level="word"`
- `skip_relational=True`
- Full-page input
- No CTD
- No PaddleOCR
- No Qwen3-VL
- No preprocessing
- No Laya

## 23.1 Aggregate result

```text
GT story blocks                  : 8
GT story blocks covered          : 8 / 8 = 100%
GT-conditioned mean CER         : 0.1990
GT-conditioned exact-match rate : 2 / 8 = 25%
Total predicted regions         : 142
Regions inside story GT boxes   : 67
Regions outside story GT boxes  : 75
Inside-story-box rate            : 47.18%
Mean runtime                     : 7,849.7 ms/page (reported benchmark mean)
```

The benchmark JSON reports mean runtime as approximately **2,876.4 ms/page** when averaged across the three page-level runtime values (22,927.6 ms, 402.14 ms, 219.46 ms). The first page includes the one-time model/backbone initialization overhead, so this average is not a steady-state per-page inference rate.

The GT-conditioned CER is computed only from predictions collected inside known story GT boxes. It must not be interpreted as end-to-end story extraction performance.

## 23.2 Per-page results

### Page 01

```text
Predicted regions                  : 49
GT story blocks                    : 3
GT blocks covered                  : 3 / 3
GT-conditioned mean CER            : 0.1715
GT-conditioned exact-match rate    : 1 / 3 = 33.3%
Regions inside story GT boxes      : 25
Regions outside story GT boxes     : 24
```

Representative outputs:

```text
GT      : HUH? DOES KOMABA SEEM DIFFERENT THAN USUAL?
Nemotron: HUH? DOES KOMABA SEEM DIFFERENT UISUL? THAN
CER     : 0.2558
```

```text
GT      : HE'S DOING REALLY WELL. IT LOOKS LIKE HE'S BEEN WORKING HARD FOR THE FALL TOURNAMENT.
Nemotron: HE'S DOING LOOKS IT REALLY HE'S LIKE WELL. BEEN WORKING HARD FOR THE FALL TOURNAMENT.
CER     : 0.2588
```

```text
GT      : OUT FOR BLOOD.
Nemotron: OUT FOR BLOOD.
CER     : 0.0000
Exact   : yes
```

### Page 02

```text
Predicted regions                  : 49
GT story blocks                    : 0
Regions inside story GT boxes      : 0
Regions outside story GT boxes     : 49
```

There are no labelled story utterance blocks on this page in the current oracle, so no recognition score is reported for page 02.

### Page 03

```text
Predicted regions                  : 44
GT story blocks                    : 5
GT blocks covered                  : 5 / 5
GT-conditioned mean CER            : 0.2155
GT-conditioned exact-match rate    : 1 / 5 = 20%
Regions inside story GT boxes      : 42
Regions outside story GT boxes     : 2
```

Representative outputs:

```text
GT      : RIGHT HERE! THIS IS WHERE IT HAPPENS!
Nemotron: RIGHT HERE! THIS IS WHERE IT HAPPENS!
CER     : 0.0000
Exact   : yes
```

```text
GT      : OKAWA-SENPAI, YOU LOOK SO COOL!
Nemotron: OKAWA- SENPAI, LOOK YOU COOL! SO
CER     : 0.3226
```

```text
GT      : IT'S ALL FOR THE SAKE OF OUR CLUB. WE'RE GOING TO ANALYZE IT IN DETAIL LATER.
Nemotron: IT'S ALL FOR SAKE OF OUR THE GOING CLUBB WE'RE TO IT IN ANALYZE LATER. DETAIL
CER     : 0.4286
```

```text
GT      : DO YOU HAVE TO KEEP WATCHING THAT? IT'S EMBARRASSING.
Nemotron: YOU DO HAVE KEEP TO WATCHING THAT? IT'S EMBAR- RASSING.
CER     : 0.2264
```

```text
GT      : SO CLOSE!!
Nemotron: SO CLOSE !!
CER     : 0.1000
```

## 23.3 What the benchmark establishes

1. **Independent page-level detection is real.** Nemotron covered all 8 labelled story blocks in this sequence when measured by the benchmark's geometric conditioning procedure.

2. **Recognition is useful but not sufficient.** GT-conditioned mean CER was `0.1990` and only 2/8 blocks were exact. The model recovered substantial text content but frequently produced incorrect word ordering because raw word regions are not assembled into the benchmark's utterance order.

3. **Raw output contains substantial non-story text.** 75 of 142 predicted word regions were outside the known story GT boxes. This is evidence that Nemotron is an OCR system, not a story-text filter.

4. **The relational layer should not be assumed to solve manga utterance grouping.** Earlier word/sentence/paragraph probes on page 01 showed continued fragmentation of the central multi-line dialogue. The standalone quantitative run therefore used `skip_relational=True` as the primary capability measurement.

5. **Runtime has a cold-start component.** Page 01 took about 22.9 s while pages 02 and 03 took about 0.40 s and 0.22 s respectively. The first page therefore should not be used alone as the steady-state throughput estimate.

## 23.4 Architecture implication

Nemotron is now empirically justified as an **independent OCR evidence source**, but not as the final manga utterance segmenter or story-text filter.

Current evidence supports:

```text
              CTD
               │
               ├──────────────┐
               │              │
           PaddleOCR      Nemotron
               │              │
               └──────┬───────┘
                      │
                Candidate Bank
                      │
                    Qwen3-VL
                      │
                     Laya
```

The exact ordering of Qwen3-VL and the OCR candidates remains governed by the locked architecture: all experts contribute structured evidence; Laya makes a typed selection rather than simply averaging confidence scores.

Nemotron should initially contribute:
- recognized text candidates,
- confidence,
- bounding-box geometry,
- optional word-level layout evidence.

Do not use Nemotron's confidence as a calibrated probability without held-out calibration.

## 23.5 Benchmark limitation

This is **one 3-page development sequence** and therefore remains a diagnostic result, not a dataset-level estimate. The next meaningful comparison is to run the same standalone benchmark across additional labelled development sequences and compare Nemotron against the existing CTD/PaddleOCR/Qwen evidence under the same geometric evaluation protocol.

## 23.6 Current gate

```text
Nemotron installation                        ✅
Nemotron sample-page probe                   ✅
Nemotron 3-page quantitative benchmark      ✅
Nemotron coverage on this sequence           ✅ 8/8
Nemotron exact recognition on this sequence  ✅ 2/8
Nemotron Candidate Bank inclusion            ⏳ cross-sequence validation
Laya training                                ⏸️
```

## 24. Nemotron 4-sequence cross-sequence validation

The standalone Nemotron OCR benchmark was extended from the initial diagnostic sequence to four labelled development sequences using the same controlled configuration:

```text
model          = nvidia/nemotron-ocr-v2
merge_level    = word
skip_relational = true
benchmark      = full-page standalone OCR with GT-conditioned geometric evaluation
```

### 24.1 Per-sequence results

| Sequence | GT blocks | Covered | Coverage | GT-conditioned CER | Exact | Predicted regions | Inside GT boxes | Outside | Mean runtime/page |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seq_2032620aa4e4ac7f | 8 | 8 | 100% | 0.1990 | 2/8 (25.0%) | 142 | 67 | 75 | 7849.7 ms |
| seq_7db7a000adeceacf | 27 | 27 | 100% | 0.2469 | 4/27 (14.8%) | 295 | 254 | 41 | 7275.6 ms |
| seq_952f154fb1505883 | 30 | 30 | 100% | 0.2370 | 5/30 (16.7%) | 283 | 267 | 16 | 7141.3 ms |
| seq_9032e5551cf1a9d2 | 26 | 26 | 100% | 0.2402 | 3/26 (11.5%) | 250 | 229 | 21 | 7063.1 ms |
| **Aggregate** | **91** | **91** | **100%** | **0.2375** | **14/91 (15.4%)** | **970** | **817** | **153** | **7332.4 ms/page*** |

`CER` aggregate is GT-block weighted. Exact-match aggregate is block weighted. Runtime aggregate is the arithmetic mean of the four reported sequence-level mean page runtimes and is not a production throughput estimate.

### 24.2 Interpretation

Across all 91 labelled story blocks, Nemotron covered every GT block under the benchmark's geometric conditioning procedure. Recognition quality remained materially below exact transcription: weighted mean CER 0.2375 and 14/91 exact matches.

The output also contains non-story detections: 153 of 970 predicted regions were outside known story GT boxes. This reinforces that Nemotron is an OCR evidence source rather than a story-text filter.

The cross-sequence pattern is sufficiently consistent to promote Nemotron from an experimental probe to an **approved Candidate Bank evidence source**, while keeping its role bounded to structured OCR/layout evidence.

### 24.3 Locked role

Nemotron contributes to the Candidate Bank:
- candidate text,
- OCR confidence,
- bounding-box geometry,
- word-level spatial/layout evidence,
- optional grouping evidence where useful.

Nemotron is **not** the final authority for:
- story-text filtering,
- utterance grouping,
- reading order,
- speaker attribution,
- cross-page character identity.

Its confidence values are not assumed to be calibrated probabilities.

### 24.4 Evidence limitation

The benchmark remains GT-conditioned. It establishes independent OCR usefulness under known story-box correspondence, not end-to-end story extraction recall/precision. No claim of test-set performance follows from these results.

### 24.5 Gate status after validation

```text
Nemotron installation                        ✅
Nemotron sample-page probe                   ✅
Nemotron 3-page quantitative benchmark      ✅
Nemotron 4-sequence validation              ✅
Candidate Bank inclusion                    ✅ LOCKED
Laya training                                ⏸️
```

### 24.6 Next implementation gate

Implement the **Candidate Bank schema and fusion adapter** so that CTD-localized blocks can receive structured evidence from PaddleOCR, Nemotron, and Qwen3-VL without prematurely collapsing candidates.

## 25. Final architecture revision: from candidate selector to multimodal adjudication

### 25.1 Why the architecture changed

The previous design used Laya as an image-free policy that selected one Candidate Bank candidate. That design was useful for forcing structured evidence and explicit candidate IDs, but it imposed an important restriction: the policy could not inspect the pixels when all candidate transcriptions were wrong or ambiguous.

The revised design removes Laya from the production path and makes the localized balloon image the primary evidence for final transcription. Candidate Bank evidence remains because independent OCR proposals are useful for difficult glyphs, punctuation, word boundaries, and cross-model disagreement.

The revised relationship is:

```text
Candidate Bank + balloon image + context
                    ↓
             Qwen3-VL adjudicator
                    ↓
             final balloon text
```

Qwen3-VL's official documentation describes expanded OCR, visual recognition, and advanced spatial perception/grounding capabilities, supporting its use as a multimodal evidence consumer rather than a text-only candidate selector. See the official Qwen3-VL repository and OCR/spatial-understanding cookbooks.

### 25.2 Research findings incorporated

Recent manga research surfaced several issues that directly affect this project architecture:

1. Manga109-v2026 documents transcription errors, missing text regions, overlapping dialogue and onomatopoeia, and under-segmented speech balloons. This supports treating balloon/utterance grouping as an explicit layer rather than assuming CTD regions are semantic utterances.

2. Manga109Dialog models speaker detection as a relationship between text, characters, and comic layout, and reports improved results from considering frame reading order. This supports a scene/graph-style speaker resolver rather than text-only attribution.

3. Recent manga VLM work shows that manga-specialized adaptation can substantially improve OCR, but the available project development set is small. The initial architecture therefore uses Qwen3-VL as a local multimodal adjudicator and reserves full VLM fine-tuning for a later, evidence-driven phase.

4. Modern manga segmentation work treats frame/panel, dialogue, onomatopoeia, balloon, face, and body as distinct semantic objects. This supports introducing explicit panel and balloon structures.

5. Reading-order research supports hierarchical/panel-aware ordering rather than a naive global coordinate sort.

### 25.3 Final semantic hierarchy

The system now distinguishes the following layers:

```text
Page
  ↓
Panel / layout region
  ↓
CTD TextRegion(s)
  ↓
Balloon / caption / utterance
  ↓
Candidate evidence
  ↓
Final balloon transcription
```

A critical invariant is:

```text
CTD region ID = provenance identity
Balloon ID     = semantic grouping identity
```

A balloon may contain multiple CTD regions.

### 25.4 Final OCR/evidence path

```text
balloon crop
    │
    ├── PaddleOCR
    ├── Nemotron OCR
    └── Qwen3-VL proposal
            │
            ▼
      Candidate Bank
            │
            ├── candidate text
            ├── confidence/evidence
            ├── geometry
            ├── preprocessing identity
            └── source metadata
            │
            ▼
      Qwen3-VL adjudicator
            │
            ├── final_text
            ├── text_type
            ├── include_in_story
            └── uncertainty metadata
```

The adjudicator is explicitly image-primary. Candidate agreement is evidence, not majority vote.

### 25.5 Final sequence reasoning path

```text
final balloon texts
       ↓
story-text classification/filtering
       ↓
panel-aware reading order
       ↓
character perception
       ↓
speaker grounding
       ↓
cross-page character identity
       ↓
sequence-level consistency
       ↓
validator
       ↓
JSONL
```

### 25.6 Gap-resolution matrix

| Gap | Planned solution |
|---|---|
| CTD fragments do not equal utterances | Balloon grouping |
| SFX/dialogue/sign ambiguity | Multimodal classification + validator |
| Complex manga layout | Panel/layout layer with geometry fallback |
| Reading order | Hierarchical panel/balloon ordering + pairwise ranker |
| Speaker attribution | Tail + spatial + visual grounding |
| Cross-page character identity | Sequence-local identity graph |
| Adjudicator hallucination | Image-primary prompt + verification/validation |
| All OCR candidates wrong | Adjudicator can generate corrected text |
| Producer/backend failure | Existing PagePerception diagnostics and LangGraph routing |
| Explicit learned component | Lightweight pairwise reading-order ranker |
| Nemotron Windows/WSL boundary | Injected local bridge |

### 25.7 Laya status

```text
Laya production dependency       ❌ REMOVED
Laya policy selection            ❌ REMOVED
Laya image-free candidate judge  ❌ REMOVED
Candidate Bank                   ✅ RETAINED
Qwen3-VL adjudicator             ✅ LOCKED
```

The previous Laya implementation remains historical code until migration cleanup; it is no longer part of the final architecture.

### 25.8 Implementation status at architecture lock

```text
CTD localization                 ✅
Deterministic crop generation   ✅
Paddle adapter                  ✅
Nemotron adapter                ✅
Qwen local runner               ✅
Candidate Bank                 ✅
Page perception                ✅
Nemotron 4-sequence validation ✅
Laya removal decision          ✅
Panel representation           ⏳
Balloon grouping               ⏳
Balloon adjudicator            ⏳
Story filter                   ⏳
Reading-order ranker           ⏳
Character perception           ⏳
Speaker grounding              ⏳
Cross-page identity            ⏳
Sequence resolver              ⏳
Validator/JSONL               ⏳
Nemotron WSL bridge            ⏳
```

### 25.9 Implementation order

The next engineering gate is **Panel/Layout + Balloon/Utterance Grouping**.

The ordering is deliberate because final transcription, story filtering, reading order, and speaker grounding all become more reliable when they operate on a true semantic balloon/utterance unit rather than raw detector fragments.

No new standalone OCR quality benchmark is required unless an implementation regression creates a specific evidence gap.

### 25.10 Research references

- Baek et al., *Manga109-v2026: Revisiting Manga109 Annotations for Modern Manga Understanding*, 2026: https://arxiv.org/abs/2605.21182
- Li, Aizawa, Matsui, *Manga109Dialog: A Large-scale Dialogue Dataset for Comics Speaker Detection*, 2023: https://arxiv.org/abs/2306.17469
- QwenLM, *Qwen3-VL* official repository, including OCR and spatial-understanding resources: https://github.com/QwenLM/Qwen3-VL

---

# 26. Integration baseline — 2026-09-29

## 26.1 Full real 3-page integration result

The complete development integration smoke was executed on:

```text
sequence_id: seq_952f154fb1505883
pages:       01.png, 02.png, 03.png
```

Reported regression status:

```text
212/212 tests passing
```

The real 3-page pipeline completed successfully with all seven backend components available and produced a valid competition-format JSONL submission.

### Real backend status

| Component | Backend | Status |
|---|---|---|
| CTD localization | `comictextdetector.pt.onnx` | ✅ |
| Manga balloon segmentation | `manga109-segmentation-bubble` YOLO | ✅ |
| PaddleOCR | `en_PP-OCRv5_mobile_rec_onnx` | ✅ |
| Qwen3-VL | `Qwen3-VL-4B-Instruct`, 4-bit | ✅ |
| Nemotron | persistent WSL worker, `v2_english` | ✅ |
| Character detection | `rtdetrv4-x-manga109s_v2` ONNX | ✅ |
| Character embeddings | MobileNetV3 local torchvision | ✅ |
| Qwen visual speaker grounding | `SpeakerContextImageStore` + `QwenVisualSpeakerGrounder` | ✅ wired |

### End-to-end counts

| Metric | Value |
|---|---:|
| Pages | 3 |
| CTD regions | 34 |
| Balloons | 29 |
| Story balloons | 29 |
| Candidates | 119 |
| Characters | 31 |
| Speaker-resolved items | 15 |
| Narration items | 14 |
| Unresolved speakers | 0 |
| Identity clusters | 12 |
| Matched identity clusters | 11 |
| Unmatched identity clusters | 1 |
| Identity pairs | 318 |
| Reading-order items | 29 |
| Resolver diagnostics | 0 |
| Validation errors | 0 |
| Serialized items | 29 |

## 26.2 Submission-contract verification

The integration produced:

```text
serialized successfully: yes
top-level keys: ['pages', 'sequence_id']
page count: 3
all speaker strings valid: True
all text strings valid: True
validation errors: 0
```

The speaker-label boundary rules remain enforced:

```text
NARRATION  ← narration / thought / sound_effect / other non-speaker story text
UNKNOWN    ← unresolved or ambiguous speaker identity
A/B/C...   ← only sequence-resolved character identities
```

No fabricated character label is introduced merely because an internal identity cluster is ambiguous.

## 26.3 Integration-stage runtime

Measured across the three pages:

| Stage | Runtime |
|---|---:|
| CTD localization | 24.35 s |
| Layout / balloon grouping | 1.34 s |
| CTD crop generation | 0.05 s |
| OCR + Qwen proposal production | 353.94 s |
| Qwen balloon adjudication | 1146.75 s |
| Character detection | 11.29 s |
| Reading order | 0.01 s |
| Character identity | 2.81 s |
| Sequence resolver | <0.01 s |

The page wall-clock times were approximately:

```text
page 0: 1758.8 s
page 1:  599.0 s
page 2: 1563.9 s
```

The measured Qwen adjudication stage is the largest explicitly instrumented cost. The current smoke script does not report a dedicated speaker-grounding timer, so the total wall-clock time is not fully explained by the stage table.

### Runtime conclusion

This integration run establishes functional completion of the pipeline, but **runtime is not yet production-ready**.

The next performance investigation should instrument and separate:

```text
Qwen proposal calls
Qwen adjudication calls
Qwen visual speaker calls
Nemotron calls
PaddleOCR calls
```

before making architectural performance changes.

## 26.4 Nemotron bridge completion

Nemotron is now integrated through a persistent WSL worker:

```text
Windows pipeline
      │
      ▼
NemotronWSLBridge
      │
      ▼
persistent Linux/WSL worker
      │
      ▼
Nemotron OCR v2
```

The worker is started once and shut down in the pipeline cleanup path. This removes repeated model initialization across pages.

Added:

```text
tools/nemotron_worker.py
app/models/nemotron_wsl_bridge.py
tests/test_nemotron_wsl_bridge.py
```

The bridge covers Windows↔WSL path conversion, request routing, error handling, startup, and clean shutdown.

## 26.5 Qwen visual speaker grounding integration

Speaker grounding now supports visual context generation for story balloons:

```text
page + characters + balloon
          ↓
SpeakerContextImageStore
          ↓
labeled visual context
          ↓
QwenVisualSpeakerGrounder
```

The architecture still prefers deterministic geometry evidence and uses visual grounding when geometry is insufficient.

Visual context artifacts are stored under:

```text
.smoke/pipeline-integration/<sequence_id>/speaker_contexts/
```

## 26.6 Current locked architecture

The implementation now exercises the complete locked flow:

```text
3-page sequence
    ↓
page perception
    ↓
CTD text localization
    ↓
panel/layout + balloon grouping
    ↓
deterministic crops
    ↓
PaddleOCR + Nemotron + Qwen evidence
    ↓
Candidate Bank
    ↓
Qwen3-VL multimodal balloon adjudication
    ↓
story-text inclusion/type
    ↓
character perception
    ↓
speaker grounding
    ↓
reading order
    ↓
cross-page character identity
    ↓
sequence consistency
    ↓
validator
    ↓
competition JSONL
```

LangGraph remains the control plane for state, routing, retries, dependency management, parallelism, and diagnostics.

## 26.7 Laya remains removed

Laya is not part of the production architecture.

```text
Laya production dependency: ❌
Laya candidate-selection policy: ❌
Image-free Laya judge: ❌
Candidate Bank: ✅
Qwen3-VL multimodal adjudicator: ✅
```

The reason for removal remains architectural: hard manga OCR cases require access to image pixels. The Qwen adjudicator is image-primary and can correct jointly-wrong candidate proposals.

## 26.8 Current engineering gate

The project has moved from:

```text
architecture implementation
        ↓
component integration
```

to:

```text
integration baseline
        ↓
development-set evaluation
        ↓
error analysis
        ↓
targeted optimization / learning
```

The next major quality milestone is automatic evaluation on the labelled development sequences using the official scoring code.

No test-set predictions should be manually edited.

