# Jevlet v7 pre-implementation audit

Date: 2026-09-26. Baseline under audit: v6, HF `NAME0x0/Jevlet` tag `v6`
(sha c59f2057ed446cfe8c57b0a041690644a11d1b05), trained from git commit 9d62c46.
Nothing in this audit trained a model, touched the vault files, or changed v6 artifacts.
All v6 error analysis below was run on **dev** splits. The vault numbers are the ones
already in `results.json`.

Evidence labels used throughout:

| Label | Meaning |
|---|---|
| **[O]** observed | Read directly from an artifact or code during this session |
| **[M]** measured | An experiment run during this session (scripts in the session scratchpad; commands reproducible) |
| **[X]** external | A third party's claim, with a note on who checked what |
| **[I]** inference | A reasoned step from [O]/[M]/[X] evidence |
| **[H]** hypothesis | A testable claim, with its falsifier |
| **[S]** speculation | Plausible, but untested and not directly supported |
| **[A]** assumed | From background knowledge, not checked this session |

---

## 0. Summary

Every headline number in the brief matches `results.json` exactly. The interpretation of
several of them does not hold up:

1. **Grounding is much worse than 78.6%.** That figure averages two questions per example:
   the control choice and a trivial risk yes/no that scores ~100%. On held-out apps with the
   target control on screen, v6 picks it **48.7%** of the time, at 0.88 mean confidence
   (ECE 0.43). In 94.6% of those errors it answers "None of these controls". [M]
2. **"OOD 77.9%" is not a separate result.** `is_ood` is set only by the grounding
   generator, so it is the same held-out-app grounding set seen through another subsample. [O]
3. **The risk "calibration got worse" anomaly is a metric mismatch, not a model defect.**
   Risk is trained on soft targets. ECE scores it against hard argmax correctness. v6
   reproduces the soft targets almost exactly (mean absolute error 0.007). The deeper issue
   is that for 46 of 50 skills the risk target is a constant per skill, so the risk head
   mostly re-encodes the skill decision. [O][M]
4. **The 98% real-command result covers half the product.** 23 of 50 skills have **no**
   real human test commands. That includes every desktop-control skill: apps, windows,
   settings, files, typing, click, power. 24.7% of real dev commands also occur verbatim
   (after normalisation) in real train. On novel dev commands skill accuracy is 97.3%, not
   98–99%. In a 40-error sample, most remaining errors are wrong gold labels or genuinely
   ambiguous. [M]
5. **`shared_state_speedup ≈ 0.99` is expected by construction and measures nothing
   useful.** The state is ~5% of the tokens and the candidate list ~84%. The benchmark times
   one example at batch 1 on an A100, which is launch-bound. The real reuse lever is the
   candidates, and in the current topology they cannot be cached because they attend to the
   state. [O][I]
6. **Pipeline and reporting defects.** `train_seconds` (and the model card's public "3.34 h")
   cover only the resumed session, steps 8,550→25,000; total A100 training was about 5 h.
   The progress log lost steps 0–8,500 on resume. The shipped C# app still loads the **v4**
   ONNX export against the 50-skill v6 catalogue. [O]

**Recommendation.** The contrastive/listwise hybrid is sound for **dynamic, high-cardinality
candidates**: apps, files, windows and UI controls. It is **not** supported as a replacement
for the 50-skill listwise pass. It also does not address grounding's measured root causes,
which are over-abstention and narrow training phrasing. Lead v7 with (1) a frozen
JevletBench-v1, (2) explicit device state plus app-diverse grounding data as the primary
hypothesis, and (3) factorized candidate caching as a secondary, compute-track hypothesis.
Test it against a lower-layer-caching variant (DeFormer-style) as well as the two-stage
cascade. Section 9 has details.

Status of the six gates before any large v7 run (updated 2026-09-26, after the owner's decisions
in §14):

| Gate | Status |
|---|---|
| 1. v6 audit complete | Done (this document). The v6 ONNX deviation was re-derived: 2.72e-5 (§1.9). The int8 trade-off was not re-run |
| 2. External research complete | Done for Jev, Kev and CLM (sections 2–4). CLM's Notion blog was unreadable |
| 3. JevletBench-v1 frozen | **Designed, awaiting owner approval** (`research/jevletbench-v1.md`). Not frozen |
| 4. v7 hypothesis explicit | Written in section 9, amended in §14 |
| 5. Falsifiable experiment | Designed in section 10 |
| 6. Baseline measurements recorded | Partly: v6 dev measurements are in this report. v6 baselines on JevletBench-v1 are recorded at freeze |

---

## 1. Current Jevlet v6 state

### 1.1 Verified metrics

Source: `data/models/v6/results.json`, `run_config.json`, `progress.jsonl`.

| Claim in brief | Artifact value | Verdict |
|---|---|---|
| Vault 23,279 commands / 61,033 questions | 23,279 / 61,033 | ✓ [O] |
| Aggregate ~98.468% | 0.98468 | ✓ [O] |
| Skill 98.0025%, ECE .00973→.00134, NLL .0757 | 0.98002, 0.00973→0.00134, 0.0757 | ✓ [O] (the NLL is after temperature) |
| Slot 97.0501%, ECE .01223→.00399, NLL .1060 | 0.97050, 0.01223→0.00399, 0.1060 | ✓ [O] |
| Risk 99.8153%, ECE .02985→.03232, NLL .1271 | 0.99815, 0.02985→0.03232, 0.1271 | ✓ values; see §1.3 for what they mean |
| Mixture dev 97.723 / skill 99.073 / slot 99.187 / risk 99.890 | same | ✓ [O]. Calibrated assistant ECEs here are in-sample (§1.3) |
| Grounding 78.633%, ECE 0.176 | 0.78633 / 0.17583 | ✓ values; wrong meaning (§1.5) |
| Banking77 96.8 / CLINC 97.73 / MASSIVE 96.73 / MNLI 72.7 / BoolQ 69.3 | same (n = 1,000 / 1,500 / 1,500 / 1,000 / 1,000) | ✓ [O]. These are in-mixture dev splits of trained datasets, not zero-shot transfer |
| OOD 77.88% | `ood_accuracy` 0.77883 | ✓ value; it is held-out-app grounding (§1.5) |
| Scaling skill 94.94/96.84/97.19/98.00, slot 92.26/94.88/95.95/97.05, risk 99.10/99.52/99.76/99.82 | same | ✓ [O]; confounded by the LR schedule (§1.8) |
| 33,508,992 params | 33,508,992 | ✓ [O] |
| 25,000 steps × 64 = 1.6M examples | 25,000, effective batch 64 | ✓ [O] |
| ~1.35M unique training rows | 1,350,835 rows | Rows ✓; "unique" ✗: ~0.55M distinct task texts (§1.7) |
| ~3.34 h total train time | `train_seconds` 12,017 | ✗ Covers only the second Colab session (steps 8,550→25,000, 11,975 s wall by log timestamps). At the logged 1.37 steps/s, 25k steps ≈ **5.0 h** of A100 time [O][I] |
| ~10.4 GB peak VRAM | 10,403.6 MB | ✓ [O]. Training at batch 64, bf16; not an inference figure |
| Temperatures 1.418 / 1.031 / 1.136 | choice 1.4181, noul 1.0307, default 1.1358 | ✓ [O] |
| shared_state_speedup ≈ 0.9877 | 0.98771 | ✓ value; not a meaningful measurement (§1.4) |
| Option-order agreement 1.0, max Δ 0.00388 | 1.0 / 0.003884 | ✓ [O], but on only **64** examples |

The data mixture composition is exactly as claimed [O]. Assistant 782,674 = 600,000 composed +
182,674 real. Daily 240,000. Grounding 150,000. Public 96,222 = MNLI 80,000 + BoolQ 7,876 +
Banking77 8,346. Intents 19,647 = CLINC 10,000 + MASSIVE 9,647. Synthetic 60,000 = six
families of about 10k each. Teacher 2,292.

### 1.2 Architecture (as implemented) [O]

- **Backbone:** BGE-small-en-v1.5, 12 layers, d=384, fully fine-tuned. Five structure tokens
  (ids 30522–30526) are added.
- **Packing:** `[CLS] state` followed by one branch per question
  (`[QUESTION] q [OPTION] o₁ [END_OPTION] … [DECIDE]`). Branch position ids restart after the
  state.
- **`block_bidir` mask** (`pretrained.py:build_pretrained_mask`): state tokens attend only to
  the state. Every branch token attends to the state plus its **own** branch,
  bidirectionally. Options therefore see the state, the question and their sibling options.
- **Pooling and head:** mean pooling over each option's tokens. Pointer head
  `key(option)·query(DECIDE)/√384`.
- **Loss:** CE + 0.25·Brier. Soft targets are used where `target_probs` is set (risk on
  assistant rows).
- **Calibration:** post-hoc temperatures per question kind, fitted by LBFGS on soft-target
  NLL over `real_commands/dev` + `commands_v6/dev` (94,505 questions).
- **Skill question:** 50 names only, ≈312 option tokens. Names plus descriptions would need
  ≈826 positions, beyond BERT's 512.

### 1.3 The risk calibration anomaly, explained [O][M]

- **How the metric is computed** (`metrics.py:compute_metrics`): ECE bins the top-probability
  confidence against **hard** correctness, `argmax == label`. For risk, `label = 0 if risk ≥
  0.5 else 1`.
- **How temperature is fitted** (`calibration.py:fit_temperature`): it minimises NLL against
  the **soft** target `[risk, 1−risk]`.
- **The mismatch:** a model that is perfect on soft targets outputs P(risky)=0.04 for a
  benign skill whose table value is 0.04. Its confidence (0.96) is then "miscalibrated"
  against a 99.8% hit rate by about 0.04. That is the ECE the brief sees. Temperature
  scaling moves toward the soft targets, so hard-label ECE rises slightly.
- **Real dev evidence** [M]:
  - The mean P(risky) of 0.0608 matches the mean target of 0.0598.
  - Mean absolute error against the soft targets is **0.0073**.
  - Binned reliability against the soft targets is on the diagonal: for example, predicted
    0.734 vs target 0.726 in the 0.5–0.8 bin.
  - Mean top confidence is 0.963 vs a mean max-target of 0.965.
  - The same arithmetic fits the mixture-dev "calibration" family (97.9% accuracy, ECE 0.150)
    [I].
- **The deeper finding:**
  - In `commands_v6` and in the real-command mapping (`real_commands.py: RISK[skill]`), risk
    is **one constant per skill** for 46 of 50 skills [O][M]. Only shortcut, click, clarify
    and power vary.
  - 4.4% of real dev commands are risky (564/12,682), so a majority-class baseline scores
    95.6% [M].
  - "Risk 99.8%" therefore mostly measures whether the skill was routed right plus a table
    lookup. It says almost nothing about recognising harm inside a skill's content: "press
    ctrl+c" vs "press shift+delete", "click Save" vs "click Send".
- **Fix:** report risk with soft-target metrics (MAE, Brier against the target, reliability
  vs target) plus a hard-decision gate metric (false-safe rate at the execution threshold).
  Do not report hard-label ECE on soft-target questions.
- **Confirmed on the vault** (2026-09-27, aggregate read of the regression guard) [M]:
  - scored against its soft targets, risk calibration error is 0.0027 raw and 0.0013 after
    temperature scaling;
  - mean target distance is 0.0064 → 0.0060;
  - hard-label ECE goes 0.0299 → 0.0323 over the same 23,279 questions.

  Temperature scaling improves the quantity it optimises. The apparent degradation is entirely the
  metric. The new evaluator also reproduces every published vault number exactly.
- **Calibration hygiene** [O]: the temperatures were fitted on `real_commands/dev` and
  `commands_v6/dev`, both of which are inside `mixture_v6/dev`. So "calibrated ECE" for the
  assistant groups on mixture dev is in-sample. The vault is clean with respect to
  calibration. However, the vault is a random split of the same TOPv2/MASSIVE/CLINC pool, so
  the vault tells us nothing about whether the temperatures transfer to new sources. Kev
  found that temperatures fitted in-distribution do not transfer [X, Kev PLAN.md:108-120].

### 1.4 Computational sharing: what `shared_state_speedup` actually measures [O][I]

**The code.** `training.py:benchmark_state_sharing` takes the **first** eval example with more
than one question. It times 5 forward passes at **batch 1** under the packed topology and
under `separate` (one row per question, state duplicated), and returns separate/packed.

**Why the measurement is uninformative:**
- The C# latency test's skill+risk pass is 372 tokens [M]. By inference, the state is about
  17 tokens, the risk branch about 28, and the skill branch about 327 (of which ~312 are
  candidate names) [I from the 312-token figure].
- Duplicating the state in `separate` therefore adds only ~5–15% more tokens.
- At batch 1 on an A100, a 12-layer 384-d encoder is kernel-launch-bound, so tokens barely
  change wall time.
- A ratio near 1.0 is guaranteed whether or not sharing works. The metric can neither
  confirm nor refute sharing.

**What Jevlet actually has:**
- **Real** computational sharing of state tokens. In a packed row they are processed once and
  every branch reads them.
- That shared part is ~5% of compute. Dense T×T attention is then masked. Attention is about
  14% of the FLOPs at T=372, so masked-out work is a few percent of the total [I, FLOP
  estimate below].

Answer to the brief's question: **both, but the shared part is too small to matter.**
Hypothesis C's "dense then mask" description is correct for attention, but that is not
where the cost is.

**Where the cost is:**
- Per pass at T=372: 12 × (372 × 1.77M MAC for QKV/O/FFN + 2 × 372² × 384 MAC for attention)
  ≈ 9.2 GMAC ≈ 18 GFLOP. About 84% of the token-linear compute is the 50 candidate names,
  re-encoded on every call [I].
- They cannot be cached: under `block_bidir` every option token attends to the state, and in
  the product the state is the command text, which changes on every keystroke.

**Structural observation** [I]: Jev and Kev optimise the regime of a long, shared state with
many questions (documents). Jevlet's workload is the inverse: a short, fast-changing state
with static or semi-static candidates. Kev-style prefix caching is therefore structurally
unimportant here. Candidate-side reuse, CLM-style or lower-layer caching, is what matters.

**An exact reuse path that already exists** [I]: `block_bidir` state tokens attend only to
the state, so the state's per-layer K/V are independent of the branches. They could be
cached exactly and reused by a later pass, for example the planner's slot pass for the same
command. This is cheap to add, but the state is short, so the gain is small.

### 1.5 Grounding, decomposed [M]

Setup: v6 run on `grounding_v6/dev`, 30,000 examples (the mixture dev uses a 3,000 subset of
the same generator). All examples are on the held-out apps **Teams** and **Spotify**.
15.3% are "foreign" tasks whose correct answer is "None of these controls". Each example has
15.5 options on average; chance is 6.5%.

| Slice | n | Accuracy | Raw ECE | Mean conf. |
|---|---|---|---|---|
| Control question, target on screen | 25,418 | **0.487** | 0.426 | 0.877 |
| Control question, foreign (answer = none) | 4,582 | 0.985 | 0.011 | 0.993 |
| Control question, all | 30,000 | 0.563 | – | – |
| Risk question (hard labels) | 30,000 | 0.9996 | ~0 | ~1.0 |
| Average of both questions | – | ≈0.781 | – | – |

The 0.781 average reproduces the reported 78.6%.

- **Per app:** control-question accuracy is Spotify 61.9% and Teams 50.7%.
- **Error types (on-screen):** of the 13,042 errors, 12,341 (94.6%) are "None of these" and
  701 are a wrong control.
- **Ranking vs abstention:** among controls only (none excluded), v6 ranks the target first
  **69.0%** of the time (top-3 87.4%). About 20 points are therefore lost to abstention and
  31 to ranking.
- **Baselines on the same on-screen items:**
  - Zero-shot BGE-small cosine (query instruction, no fine-tuning, no none option): top-1
    58.6%, top-3 76.9%.
  - Word-overlap heuristic: 36.7%.
  - Fine-tuning improved ranking by ~10 points over zero-shot; it did not destroy it.
- **Thresholding:** the best threshold on P(none) (0.99) raises the overall control score only
  to 67.9% (on-screen 63.2%, foreign drops to 93.7%). Abstention bias is not a scale problem
  a temperature or threshold can fix.
- **Cause, from the data** [M]:
  - The 150k grounding training rows contain only **~1,950 distinct task phrasings** (1.3%
    unique after normalisation) and 21% unique states.
  - The model has learned "phrase I have seen → its control" and "unfamiliar phrase → none",
    confidently.
  - The foreign rate is 15%, and foreign tasks are phrases belonging to *other known apps*.
    So "unfamiliar vocabulary ⇒ none" is a learnable shortcut [I].
- **History:** the lab notebook reported the same failure for v3 ("correct control 45–56%").
  The aggregate metric has hidden it for three versions [O]. The v3→v4 "unseen-app grounding
  77.6→82.4%" row used the same mixed metric [O].
- **Test design is too thin:** 2 held-out apps with hand-written task phrases is a
  high-variance, low-coverage OOD estimate [I].
- **Not in the product:** grounding and click are not implemented in the C# app at all [O].
  Its practical impact today is zero; it is the largest capability gap.

### 1.6 Real human commands: what the 98% measures [M]

Run on `real_commands/dev` (12,682 commands); the vault itself was not read.

- **Coverage:**
  - 27 of 50 skills appear; 23 are absent. The absent ones are open app, switch/close/
    minimize/maximize window, brightness, Wi-Fi/Bluetooth, settings page, dark mode, website,
    folder, file, type text, shortcut, click, reschedule event, take/show notes, battery,
    screenshot, lock, power, and ask AI.
  - The top 5 skills (directions, weather, alarm, reminder, music) are 54% of rows.
  - This is a phone-assistant test set, not a desktop-control test set.
- **By source:** TOPv2 11,134 rows at 98.9%, MASSIVE 1,143 at **88.4%**, CLINC 405 at 94.3%.
- **Overlap:** 24.7% of dev commands (TOPv2 27.4%) occur verbatim in real train after
  normalisation (case, punctuation, digits). Skill accuracy is **99.46% on seen vs 97.27% on
  novel**; slot 98.4% vs 97.1%. The vault is a random split built the same way, so a similar
  overlap is expected there [I; not measured, to keep the vault untouched].
- **Error audit** (random 40 of 278 skill errors, judged by me as a single annotator; treat
  as indicative):
  - About 11 have wrong gold labels from the dataset→skill mapping. Examples: "what is the
    time in london" → date/time; "open clock" → set alarm; "Exit Spotify" → pause media;
    "remove last played song" → complete a to-do.
  - About 18 are genuinely ambiguous. Roughly 10 of them depend on **device state the input
    doesn't carry**: is an alarm ringing, a timer running, media playing? Examples: "give me
    another 10 minutes", "no need for the alarm", "halt alarms".
  - About 11 are clear model errors. That implies a clear-model-error rate of about 0.6% [I].
  - Among the 15 highest-confidence errors (p≥0.99), most are label errors.
- **Selective prediction** (skill, calibrated): coverage 91.3% at ≤0.5% error, 96.8% at ≤1%,
  99.7% at ≤2%.

### 1.7 Data diversity [M]

Measured over all 1,350,835 train rows by normalising the first state line (the task).

| Family | Rows | Unique states | Unique normalised task text |
|---|---|---|---|
| assistant | 782,674 | 89.4% | 53.8% (~421k) |
| daily_synthetic | 240,000 | 56.9% | 9.3% (~22k) |
| grounding | 150,000 | 21.1% | **1.3% (~1.95k)** |
| public_mnli | 80,000 | 80.2% | 80.1% |
| contradiction / missing / permutation / semantic / calibration | ~10k each | ≤0.8% | ≤0.1% (≈10 states each, repeated ~1,000×) |
| rules | 9,986 | 53.9% | ~0 |
| CLINC / MASSIVE / Banking77 | 10k / 9.6k / 8.3k | 100% | ~99% |
| teacher | 2,292 | 16.6% | 8.3% |

The "10×" was mostly surface variation and upsampling. There are about 0.55M distinct task
texts. The five tiny synthetic families score 100% on dev because they are memorised [I].
Label balance for assistant skills is reasonable: 1.5–4.6% per skill.

**Data provenance correction (2026-09-27)** [M]:
- **The table above describes the local build, not the trained one.** It was measured on the
  local `data/mixture_v6`, which was built with personal inputs: live UIA captures of this
  machine's apps (13,539 grounding rows labelled `live:ChatGPT` / `live:WindowsTerminal`) and the
  installed-app list. The Colab training build used `personal_inputs: false`.
- **Which sources differ:** comparing source hashes in the published manifest with the local one,
  only three differ: commands train, commands dev, and grounding train. Real commands (train, dev
  and vault), grounding dev, daily, public, intents, synthetic and teacher are all byte-identical.
- **Trained sources regenerated:** the non-personal command source (seed 606, no local apps) and
  grounding source (seed 4242, no extra apps) were regenerated and match the published training
  manifest **exactly** (`13712a6e…`, `e08792d5…`).
- **Conclusion:** the public v6 weights were not trained on the local captures.
- **Diversity on the trained sources:**
  - the assistant family has 54.0% unique normalised tasks (422,264 of 782,674; the local build
    gave 53.8%);
  - the trained grounding source has **1,993 distinct normalised task phrasings** in 300,000 rows
    (0.66%), and the mixture samples 150,000 of them.

  The grounding-diversity diagnosis (§1.5) therefore holds for the data v6 actually saw.

### 1.8 Scaling and training curve [O][I]

- **The snapshots are one run:** they come from one cosine-schedule run (steps 2,500, 6,250,
  12,500), not from separate runs with full schedules. The final point benefits from LR
  annealing to 5%; the others do not. The curve conflates data seen with annealing, so it is
  not a data-scaling law [I].
- **Error rates across the snapshots:**
  - Skill error went 5.06% → 3.16% → 2.81% → 2.00%. The last doubling cut error by 29%
    relative.
  - Slot error went 7.74 → 5.12 → 4.05 → 2.95.
  - Risk ECE rose over training (0.017 → 0.028 at 12.5k), consistent with the soft-target
    mismatch in §1.3.
- **Loss:**
  - The log covers only steps 8,550–25,000; steps 0–8,500 were lost on resume.
  - Mean train loss per 2.5k window fell 0.0996 → 0.0999 → 0.0951 → 0.0838 → 0.0792 →
    0.0747 → 0.0705. It was still falling.
  - The soft risk targets set a non-zero entropy floor on this loss.
- **Would another doubling pay?** On the real-command metric, probably no more than ~0.3–0.5
  points [I]:
  - About 0.6% of dev errors are clear model errors; the rest is label noise or ambiguity.
  - 25% of the test set is already memorisable.
  - The measuring instrument is near saturation.
  - More of the same data will not move desktop-control skills (absent from the real test)
    or grounding (low phrase diversity).

### 1.9 Native runtime [O][M]

- **Loaded model:** `Settings.ResolveModelDirectory` → newest `data/models/onnx/*`, which is
  **v4-opt** (`source_checkpoint: current.pt`). There is no v6 ONNX export. The catalogue is
  `shared/skills.json` v6 (50 skills), and nothing checks that the model and catalogue
  versions match. The daily driver is asking v4 a 50-option question it was never trained
  on.
- **Parity:** 6/6 golden tests pass [M]. Tokenizer ids, packer rows, text rules and slot
  options match exactly. ONNX logits match PyTorch within the test's 2e-3 bound. The brief's
  4.2e-5 maximum deviation was **not** re-derived (the test prints no value). For v6 it is now
  measured: 2.72e-5 (see the update below).
- **Latency** (DirectML on the RTX A2000 laptop, v4 weights; the architecture is the same as
  v6, so it transfers [I]):
  - Skill+risk pass (372 tokens): GPU median 35.9 ms (p90 41.9); CPU median 221.4 ms (p90
    239.4) [M].
  - The brief's 31 ms / 226 ms are close.
- **CPU planning:**
  - `TextRules.Shortlist` over 333 synthetic app names takes **12.3 ms per call**, failing the
    test's own 10 ms budget [M].
  - Spans take 0.68 ms and time parsing 0.015 ms [M].
- **Int8:** "11% of top choices changed for ~23% speed" is recorded only in an earlier
  session. The `v4-int8` export exists, but the figure was not re-run [O].
- **Clean enough for long-term work?** Yes, in structure [I]:
  - One shared catalogue and rules JSON with golden parity tests.
  - No Python at runtime.
  - Shape bucketing, warm-up and keep-warm, device-lost recovery.
- **Gaps:**
  - v6 is not exported.
  - The skill names are re-tokenised on every call. That is small, but it is static work.
  - There is no candidate cache.
  - The app shortlist is O(apps × tokens) per keystroke.
  - There is no grounding/UIA path.
  - An earlier session saw 200–340 ms palette latency. That is not explained by the 36 ms GPU
    pass plus the 12 ms shortlist, and the end-to-end in-app breakdown is still unmeasured.
  - The C# app is **uncommitted** (git status shows the whole App project as untracked or
    modified).

**Update 2026-09-26 (step 0)** [M]:
- **v6 export:** v6 exported to `data/models/onnx/v6`. Max |ONNX − PyTorch| logit difference is
  **2.72e-5** over 111 golden cases.
- **Catalogue identity:** `model.json` now carries a catalogue identity (version 6, fingerprint
  `247e1e60…`, the 50 skill names).
  - The C# `Catalogue.Fingerprint` reproduces Python's `catalogue_fingerprint`, checked by a
    golden test.
  - The app refuses a model trained for another catalogue, and dev-export resolution skips such
    models. v4 exports are now refused.
- **Train/serve skew check:** text rules at the v6 data commit (26e7f28) and today give identical
  courtesy stripping, spans, time/duration parsing and shortlists on 32,682 real and composed
  commands, so there is no skew.
- **Shortlist speed:** caching per-name features cut `TextRules.Shortlist` over 333 apps from
  12.3 ms to **1.56 ms**. The golden parity tests still match exactly.
- **v6 latency, skill+risk pass (372 tokens):** GPU p50 34.1 ms (p90 40.4); CPU p50 233.1 ms (p90
  248.4).

### 1.10 Pipeline defects found [O]

Status: items 1–4 were fixed on 2026-09-26.
- `progress.jsonl` is restored from Drive on resume, with rows past the resume step dropped.
- Whole-run totals (`train_seconds`, `train_seconds_complete`, `training_sessions`, peak VRAM) are
  carried in the checkpoint.
- Evaluation groups questions by role (`grounding/control` vs `grounding/risk`).
- `target_ece` and `target_distance` are reported for soft-target questions.
- `ood_accuracy` was renamed `held_out_domain_accuracy`.
- The model card generator uses the new fields.
- The published v6 card is corrected only after owner approval (§14).

1. `colab.py` resume does not restore `progress.jsonl` from Drive before appending, so the
   mirror overwrote steps 0–8,500. Google Drive's file version history may still hold the
   first session's copy (30-day retention [A]).
2. `train_seconds`, `train_*_per_second` and `peak_vram_mb` are per session, not per run.
   The public model card shows "3.34 h".
3. `evaluate_checkpoint` groups the grounding family across both questions; see §1.5.
4. `is_ood` is only ever grounding, but it is reported as "OOD accuracy".
5. The scaling snapshots are evaluated without their own temperatures. Only accuracy is
   comparable across them.
6. The option-order test uses 64 examples, too few to bound a 1% flip rate.

### 1.11 Strengths and weaknesses

**Strengths** [O][M]:
- Routing on trained phone-assistant intents is fast and well calibrated.
- Temperatures fix skill/slot ECE (vault 0.0013 / 0.0040).
- Selective prediction is useful: 96.8% coverage at ≤1% error.
- The architecture is small: 33.5M params, ~18 GFLOP per pass, 36 ms on a laptop GPU.
- Python/C# parity is exact.
- The Colab path is crash-tolerant (it survived a real mid-run crash).
- Data generation is reproducible, with provenance and hashes.

**Weaknesses, by evidence strength:**
- Unseen-app grounding (48.7%, confidently wrong).
- No real test data for 23 desktop skills.
- The state carries no device state.
- Risk is a per-skill constant.
- The measurement layer hides failures (§1.3–1.5).
- The app runs v4.
- Candidate compute cannot be reused.

---

## 2. Jev update (TypeSafe)

Sources: TypeSafe docs and launch post, JevBench, and the black-box studies by Hume and
agrogov. The research agent fetched them and re-checked the key claims against raw files
where possible. All entries are **[X]** unless marked.

**What TypeSafe itself says:**
- Every question sees the same state and is evaluated independently; adding questions
  "barely changes the response time".
- Choice takes at most 255 options, and "adding options costs a few tokens each".
- For bigger taxonomies the docs say to chain Choices level by level.
- For high-cardinality Choice, the launch post describes "a 2 stage-system of scoring
  independently then making an explicit choice". The cut-over point is not given.
- Score levels are "evaluated separately… judged on its own against the state", which is
  pointwise.
- Noul answers carry no confidence field.
- Trained with "RLCD", with probabilities "optimized against outcomes". No paper, no
  parameter count, and no mention of proper scoring rules.
- The docs list known failure modes: arithmetic, dates, and degradation as unrelated state
  grows. They also say Noul and Choice framings are not mutually consistent.

**Architecture evidence:**

| Hypothesis | Confidence | Evidence |
|---|---|---|
| Probabilities read out directly, not generated | Confirmed (vendor), strongly supported | Latency flat from 1 to 128 questions |
| Shared state, encoded once | Sharing confirmed; prefix/KV reuse strongly supported | Latency scales with state length, not question count |
| Questions isolated | Confirmed | A sibling-question secret gives P=0.00; in the state, P≈0.9 (Hume) |
| Listwise interaction within Choice | Strongly supported | An irrelevant 5th option shifts the log-odds between the other two by −0.28 (95% CI −0.36…−0.19, 10/10 blocks). A 30-candidate listwise Choice beats 30 Nouls (top-10 85% vs 65%) |
| Option-order sensitivity | Present but disputed | 0–5% flips across audits. Jev is non-deterministic: 50 identical calls gave 15 distinct answers |
| Pointer/`<decide>` readout | Speculative | It is Kev's design, not observed in Jev |
| Two-stage high-cardinality Choice | Stated by the vendor; mechanism unobserved | – |
| Calibration | Mixed | Noul probability MAE 0.079. Choice overconfident: 3-class top-prob 0.93 vs 0.66 exact; a fair die gets 83% top-prob at 19% accuracy |

**Benchmarks:**
- JevBench v1.4.2: 86.6% public, **36.7% sealed** (chance 29.3%), calibration ECE 0.22 on
  sealed.
- jevbench.xyz, zero-shot Jev: Banking77 80.3%, MASSIVE 79.3%.
- JevOut: natural context additions redirect 61.4% of correct decisions.

**Lessons for Jevlet** [I]:
1. Jev itself is listwise within Choice and pointwise for Score. Mixing interaction regimes by
   question type is legitimate.
2. Jev's own two-stage claim for high cardinality supports a cascade **only** where
   cardinality is high.
3. Jev's documented overconfidence on Choice suggests calibrated abstention is a real
   differentiator for a small local model.
4. Jevlet's in-domain 96.7% (MASSIVE) and 96.8% (Banking77) against Jev's zero-shot 79–80% is
   **not** a like-for-like comparison. Jevlet trained on those datasets.

---

## 3. Kev (jaredpalmer/kev)

The repo was cloned at HEAD f1535963. Its `verify_claims.py` passed on 760 claims, which only
proves the docs match the committed JSON. No model was re-run. **[X]** unless marked.

- **Architecture:**
  - Qwen3.5/3.8 bases (0.8B–27B), LoRA r=16 plus a pointer head:
    `(W_k h_</opt>)·(W_q h_<decide>)/√256`.
  - Delimiters reuse existing Qwen special tokens (training new embeddings was rejected).
  - Options are represented by their `</opt>` hidden state.
  - Question positions restart after the state.
  - User text is escaped so it cannot forge delimiters.
  - Exact option isolation was tried and rejected (−5.8 pp at 4B).
- **Serving reuse** (confirmed in code):
  - On hybrid Gated-DeltaNet backbones, each question is its own row over a cached state
    prefix (KV plus DeltaNet state), with an LRU prefix cache in `serve.py`.
  - The parity test with a real 0.8B model is **not in CI**.
  - Published numbers use the non-cached path.
- **Training:**
  - The base set is 10k public records plus ~2.6k policy and rule items. Hard labels or soft
    targets with CE; Brier, focal and label smoothing were rejected against a matched
    control.
  - Augmentation: always-permuted options; a none-of-the-above swap (10%); none as a wrong
    option (12%); irrelevant distractors (15%); none minimal pairs on 25% of Choice records.
  - Stacking sequential deltas eroded earlier gains at 0.8B, while training jointly passed.
    Replay helped at 4B but not at 9B or 27B.
- **Calibration:**
  - One NLL temperature. The shipped temperatures were fitted on in-distribution dev rows,
    and the project now calls that a mistake: an SFT-dev fit gave breadth ECE 0.059 against
    0.0085 from a held-out-dataset fit.
  - Per-(type, K) temperatures and a logistic reliability head both made things **worse**.
- **Evaluation discipline** (the strongest part):
  - Pre-registered rounds; dev selects; test and locked sets are read once, with code that
    refuses a re-read.
  - Paired, record-clustered bootstrap (2,000 resamples).
  - Guards sized to what a panel can resolve (±1.7 pp at n=656); small panels are pooled.
  - Full negative-result logs.
- **Results:**
  - transfer-v4 locked: 0.697 (0.8B) to 0.896 (27B). The widely quoted "27B 0.848 vs Jev
    0.857" is **dev**, not locked; Jev never had a locked read.
  - Tests read 2–5 pp above dev.
  - Nothing above 10 options was evaluated.
  - Flip rates: 0.08 (0.8B, 4B) and 0.03 (9B).
- **Lessons** [I]:
  - Adopt the evaluation rules, the held-out-source temperature rule, abstention data
    (none swaps, minimal pairs, "unknowable" uniform targets), and delimiter escaping.
    Retrain jointly on the union of data rather than stacking deltas.
  - Skip the prefix-cache machinery: wrong regime for Jevlet (§1.4). Skip long context
    (BGE's 512 limit), LoRA (full fine-tuning is fine at 33M) and knowledge benchmarks.
  - Do **not** expect a reliability head to beat one well-fitted temperature without evidence.
    Kev's negative result is directly relevant to the brief's Phase 12.

---

## 4. CLM (Contrastive-LM/CLM, CLM-8B)

The repo was cloned, and both released head checkpoints were downloaded and loaded. **[X]**
unless marked.

- **Architecture** (confirmed in code and weights):
  - Frozen Qwen3-8B behind a vLLM pooling server, with last-token pooling.
  - The input is L2-normalised. Two untied MLP heads (4096→1536→1536→512, GELU, LayerNorm)
    of 9.44M params each, 18.9M total.
  - Score = scale × cosine.
  - **The learned scale is clamped at exp(4.61)=100.8→100.** Past the clamp there is no
    gradient, so the temperature is frozen at 0.01. The fine-tuned DeepSWE head has a
    bit-identical value.
- **Caching:**
  - Action and state vectors are both cached in a device arena keyed by (head generation,
    text). The state key includes the question instructions.
  - New state ≈ 28 ms on a 4090, flat from 3 to 50 actions. A revisited state takes
    1.7→0.6 ms.
- **Training code:**
  - Bidirectional in-batch InfoNCE, with same-(task, step) duplicate positives masked to −∞.
  - **No hard-negative, curriculum or replay code.**
  - The 60M/30M/1M curriculum, the hard-negative ablation (52.1→69.2% top-1) and the replay
    result (69→68.5% vs 56.2%) are **README-only**, with no data.
  - The README's fine-tune command likely does not reproduce the released DeepSWE head,
    which used a different, filtered dataset.
- **Claims:**
  - "Up to 9× faster than Jev" comes from one T-Rex run: localhost 4090 with a warm cache vs
    the hosted Jev API on a different day.
  - Both "survived 5/5" only because a harness shield replaced unsafe answers: 29% of CLM's
    decisions vs 0.5% of Jev's.
  - Planner agreement: CLM 65.8% vs Jev 98.7%.
  - Chart-only numbers show CLM **worse** on BFCL (95.2 vs 99.2) and WikiRacing (26/30 vs
    30/30).
  - DeepSWE verifier: 31/38 vs 27/38, a 4-task gap on 13 decidable tasks, fine-tuned CLM vs
    zero-shot Jev.
- **Other gaps:**
  - No abstention output.
  - Candidate vectors are independent of state and question.
  - A third-party issue reports Score collapse.
- **Autoresearch:**
  - Only `train/finetune.py` is mutable; data and eval are frozen; one change per commit;
    keep or revert.
  - **Flaw:** keep/revert is decided on the reported held-out best-of-N rate, which is
    selection on test.
- **Lessons** [I]:
  - Candidate caching makes cost flat in candidate count. That is exactly Jevlet's regime
    for apps, files and controls.
  - Mask duplicate positives. Never clamp an exponentiated scale.
  - Copy the one-mutable-module autoresearch rule, but keep/revert on **dev**.
  - Treat the curriculum and hard-negative claims as hypotheses, not evidence.
  - Pure factorization loses what Jevlet has: state-conditioned candidates, set-relative
    choices ("the other window"), near-duplicate discrimination, and learned abstention.
    Replacing Jevlet with BGE-small plus two heads would produce a "CLM-33M" and discard
    its differentiator.

---

## 5. Direct comparison

Metrics from different distributions are **not** comparable. The bottom rows say what each
number measures.

| Axis | Jev (TypeSafe) | Kev | CLM-8B | Jevlet v6 |
|---|---|---|---|---|
| Access / licence | Hosted API, proprietary | Apache-2.0 LoRA adapters | Apache-2.0 code + heads | MIT code + weights |
| Backbone | Undisclosed (causal decoder suspected) [X/S] | Qwen3.5/3.8 0.8B–27B | Frozen Qwen3-8B | BGE-small-en-v1.5, fully fine-tuned |
| Params (total / trained) | Unknown | 0.8B / 11.3M … 27B / ~117M [I] | 8B frozen + 18.9M heads | 33.5M / 33.5M |
| Compute per decision (short state) | Unknown | ~0.3–0.6 TFLOP at 0.8B for ~370 tokens [I] | ≈2×8B×tokens(state+instr), ~0.6 TFLOP at ~40 tokens; candidates cached [I] | ~18 GFLOP per 372-token pass [I] |
| Objective | "RLCD", outcome-optimised (undisclosed) | CE, soft targets | Bidirectional InfoNCE | CE + 0.25·Brier, soft risk targets |
| Data | "Synthetic" (undisclosed) | ~12.6k (base) to ~199k (SFT) records | 62.5M pretrain rows (curriculum claimed) | 1.35M rows, ~0.55M distinct tasks |
| Negatives / hard negatives | Unknown | None swaps, distractors, minimal pairs | In-batch; hard negatives claimed (Gemini), no code | Implicit (catalogue siblings, on-screen distractors); none mined |
| Calibration | Claimed; mixed in audits; Choice overconfident | One NLL temperature (now held-out datasets) | Temperature knob; no abstention | Per-kind temperature on in-distribution dev |
| Uncertainty / abstention | Noul P(yes); no Noul confidence | None options, unknowable items | None | Clarify skill, "none of these" option, confirm gate |
| State representation | Up to 32k tokens + longest question | Flattened JSON/text, restarted positions | Text + instructions → one vector | "Task: … / Active window: …" (≤224 tokens) |
| Option representation | Names + descriptions | `</opt>` hidden state | Independent vector per candidate | Mean-pooled tokens, state-conditioned |
| State reuse | Yes (confirmed sharing) | Prefix KV cache (serving path only) | State-vector cache | State tokens once per packed row (~5% of compute) |
| Option reuse / candidate cache | No known | No | **Yes** | No (options attend to the state) |
| Question isolation | Yes | Yes (mask or rows) | By construction | Yes (block mask) |
| Option interaction / listwise | Yes within Choice | Yes (causal) | No | Yes (bidirectional) |
| Permutation sensitivity | 0–5% flips (disputed) | 3–8% flips | Invariant | 0/64 flips (under-sampled) |
| High cardinality | ≤255; two-stage | ≤255 API; evaluated ≤10 | ~1k claimed, flat cost | 50 names ≈ positional limit; apps via lexical shortlist |
| Long context | 64k | 64k serving; trained ≤384 (27B: 7.5k) | 2k serving | 224-token state |
| Latency | 70–500 ms hosted; p50 0.65 s (JevBench) | 18 ms (4B, H100) | ~28 ms new state (4090); ms when cached | 36 ms GPU / 221 ms CPU on a laptop [M] |
| Memory | n/a | ~1.6 GB weights at 0.8B bf16 [I] | ~16 GB encoder [I] | ~130 MB weights fp32 [I] |
| OOD / generalisation evidence | JevBench sealed 36.7% (chance 29.3%) | Locked new-source 0.697–0.896 | Chart only | No new-source test; unseen-app grounding 48.7% |
| Knowledge / reasoning | MMLU-Pro ~83–85% (third party) | MMLU-Pro 0.23 (0.8B) – 0.665 (27B) | – | MNLI 72.7, BoolQ 69.3 (trained, in-mixture dev) |
| Agent / verifier / desktop use | General API | Decisions | Verifier, games, tool calls | Local desktop command palette |
| Personalisation / continual learning | Deployed models frozen | Deltas with replay | Replay (claimed) | Python feedback/personalise path exists; not wired to the C# app [O] |
| Safety mechanism | Guidance by stakes | – | Harness shield in demos | Per-skill risk floors, risk Noul, confirm gate, capability-limited catalogue |
| Reproducibility | None | High | Partial (key data unreleased) | High for code and data; one log partly lost |
| Benchmark quality | LLM-labelled workflow evals | Strong protocol, small panels | One confounded demo | Self-authored; dev ≈ train distribution; no locked new-source set |

**What the headline numbers measure:**
- **Jevlet 98.0% (vault):** in-distribution held-out split of three trained intent datasets,
  mapped to 27 skills, with ~25% verbatim overlap. Measures fit to trained phone-assistant
  intents.
- **Kev 0.697–0.896 (transfer-v4 locked):** zero-shot decisions on 656 questions from
  **never-trained sources**, including knowledge (MMLU). Measures transfer and knowledge.
- **Jev 86.6% / 36.7% (JevBench):** mixed public decisions / a deliberately hard sealed set.
- **CLM:** agent task success with harness assistance.

None of these can be ranked against the others. Only a shared frozen benchmark (§11.4) can
rank them.

### 5A. The brief's hypotheses A–E

**A. "v6 has largely solved command→skill routing in its domain." — Partly true; the scope is
narrower than claimed.**
- *Supported:* on trained phone-assistant intents, clear model errors are ~0.6% and most
  residual errors are label noise or ambiguity [M/I]. More generic intent-classification data
  is low-value, and the vault can no longer measure improvements.
- *Not supported:*
  - The desktop-control half of the catalogue (23 skills) has no real human test at all.
  - Benchmark v2 (98 author-written cases, 95% CI ≈ ±5 pp) misses on window management
    ("bring whatsapp to the front", "make chrome take up the whole screen") [M].
  - ~25% of residual real errors need device state, and the input doesn't carry it.
- **Conclusion:** routing of *trained intents* is saturated. Routing of *desktop commands*
  and *state-dependent commands* is unmeasured.

**B. "Grounding is the most important practical weakness." — Supported, and worse than
stated, but for other reasons.**
- True unseen-app control selection is 48.7% (on screen), with confident abstention.
- The root causes are (i) ~1.95k training phrasings, (ii) an abstention shortcut, and (iii) a
  2-app test.
- Architecture is not the leading suspect: v6 ranking without none is 69%, above zero-shot
  cosine at 58.6%.
- "Practical" is qualified, because grounding is not in the shipped app yet.

**C. "Logical parallelism, no real reuse." — Partly correct; the cited evidence doesn't test
it.**
- State tokens are shared for real, but they are ~5% of compute. Masked attention wastes a few
  percent.
- 0.988 comes from a launch-bound batch-1 timing that could not show a difference.
- The substantive point is different: candidate encoding (~84%) is recomputed on every call
  and is uncacheable under the current topology.

**D. "General knowledge/reasoning is limited; does it matter?" — Limited, yes. It matters only
through calibration.**
- MNLI 72.7 and BoolQ 69.3 are in-mixture dev (trained). BoolQ's majority baseline is ~62%
  [A].
- "OOD 77.9" is grounding, not reasoning.
- For a desktop router that can escalate, what matters is *knowing when it doesn't know*.
  v6 fails at exactly that on grounding: it is confidently "none".
- A test for later: ablate the 88k MNLI/BoolQ rows. If desktop metrics don't drop, they are
  spending capacity for nothing [H].

**E. "Don't scale up; the minimum-compute question is differentiated and publishable." —
Directionally right, but novelty must be earned.**
- Small-encoder intent routing, cascades, early exit, late interaction and selective
  classification are all established areas [A].
- What is differentiated [I]:
  - (1) Jevlet's regime: short, per-keystroke state; static candidates; local; safety-gated.
    That is the *inverse* of Jev's long-state regime, and it makes candidate-side reuse,
    not state-side reuse, the key.
  - (2) A measured compute–accuracy–calibration–unsafe-autonomy frontier against Kev and CLM
    on one frozen benchmark.
  - (3) State-conditioned minimal pairs showing when explicit state matters.
- Without (2) and baselines, it is an engineering report, not a paper.

---

## 6. Jevlet bottlenecks, ranked by evidence

| Rank | Bottleneck | Evidence | Practical impact now |
|---|---|---|---|
| 1 | **The measurement layer hides failures.** Mixed-question family scores, hard-label ECE on soft targets, OOD = grounding, in-sample calibration, schedule-confounded scaling, 98-case self-authored benchmark, ~25% train overlap, label noise | Strong [O][M] | Every v7 decision depends on it |
| 2 | **The app runs v4**, and there is no v6 export or catalogue/model version check | Strong [O] | Daily driver quality now |
| 3 | **Unseen-app grounding:** 48.7% on screen, ECE 0.43, confident "none"; 1.95k phrasings; 2 test apps | Strong [M] | Blocks click and UI automation |
| 4 | **State poverty:** no timers, alarms, media, selection, open windows or recent actions in the state | Moderate [M sample + I] | ~25% of residual real errors; the core System-One premise |
| 5 | **Desktop-command coverage:** 23 skills with no real test; window-management misses | Moderate [M] | The product's main use |
| 6 | **Risk is a per-skill constant:** no content-level harm judgement; safety tested on 6–7 cases | Strong on mechanism [O]; untested on outcomes | Safety claims unsupported |
| 7 | **Candidate compute not reusable:** 84% of the pass; CPU 221 ms; 12 ms lexical shortlist; 512-position cap on the catalogue | Strong analysis [O][I] | Moderate (the GPU path is fine) |
| 8 | Pipeline defects (§1.10) | Strong [O] | Low |
| 9 | General reasoning | Weak relevance [I] | Low |

---

## 7. What should NOT change (freeze)

1. **The v6 checkpoint** as the reproducible baseline: HF tag `v6`, sha above, git 9d62c46,
   data hashes in `results.json`. Every v7 variant is compared with it using paired tests.
2. **The packed `block_bidir` listwise pointer model for the fixed skill question.** It is
   accurate on trained intents, well calibrated after temperature, 36 ms on GPU, and
   order-stable on the small test. There is no evidence that a cascade improves it.
3. **Per-kind temperature scaling for skill and slot.** Change only the *fitting data*: use
   held-out sources.
4. **The shared catalogue and text-rule JSON and the C# golden parity tests.** They are the
   single source of truth across languages.
5. **The span-extraction rules** (85% recall on real TOPv2), until a learned extractor beats
   them on a locked set.
6. **The Colab driver's crash/resume design.** Only fix the progress-log restore and
   per-run timing.
7. **The real-command vault as a regression set.** It stays a guard ("does not drop by more
   than X"), not a target.

---

## 8. Candidate v7 directions

Each direction lists benefit, cost, novelty, risk, failure modes and a falsifier. Compute
costs assume Colab A100 (~5 h for a v6-sized run [I]).

**D1 — Measurement first: JevletBench-v1 plus metric fixes.**
- *Benefit:* makes every other result interpretable. It is a prerequisite.
- *Complexity:* low code, moderate labelling (human-verified gold, acceptable-answer sets).
- *Novelty:* low alone; enables the paper.
- *Risk:* labelling effort and single-annotator bias.
- *Compute:* none.
- *Failure mode:* a benchmark that cannot resolve v4 vs v6.
- *Falsifier:* if the paired bootstrap cannot separate v4 from v6 on the command panels, the
  panels are too small or too easy.

**D2 — Grounding data and abstention redesign.**
- *What:* app-diverse corpus (≥200 procedurally built or recorded apps; control vocabularies
  from UIA naming patterns); app-disjoint splits with ≥20 held-out apps; paraphrase coverage
  per control; foreign tasks drawn from *unseen* vocabularies so that "unfamiliar ⇒ none"
  stops being a shortcut; none minimal pairs (Kev); calibrated abstention.
- *Benefit:* targets bottleneck 3 directly.
- *Complexity:* moderate.
- *Novelty:* low to moderate.
- *Risk:* synthetic apps may not transfer to real UIA trees.
- *Compute:* one v6-sized run.
- *Failure modes:* the synthetic-to-real gap; label ambiguity between near-duplicate controls.
- *Falsifier:* if held-out-app on-screen accuracy stays below 70% with 100× more phrase
  diversity at matched steps, data is not the bottleneck; then test capacity (D8) and the
  retrieval stage (D4).

**D3 — Explicit device state.**
- *What:* add structured state lines, for example "Alarm ringing: 07:00 Morning",
  "Timer running: 4 min left", "Media: Spotify playing", "Selection: 3 files",
  "Open windows: …". Train on state-conditioned minimal pairs: same command, different
  state, different gold.
- *Benefit:* resolves ~25% of residual real errors; restores the System-One premise that
  decisions depend on state.
- *Complexity:* low to moderate (state capture in C#, generator changes).
- *Novelty:* moderate. A clean demonstration that small models exploit explicit state is
  paper-worthy.
- *Risk:* longer states eat the 512-position budget. The skill branch is already near the
  limit, which D4/D5 would relieve.
- *Compute:* one run.
- *Failure mode:* the model ignores state lines when the command is lexically strong.
- *Falsifier:* on a frozen minimal-pair panel, v6 should score near the marginal rate (it
  cannot see state). If v7 is not ≥95% there, the approach failed.
- *Constraint (2026-09-26, §14):* only fields the production app observes reliably are allowed.
  These are the tier A/B fields of the observability spec (`jevletbench-v1.md` §8). No privileged
  or convenience fields. Unobservable facts (for example, unsaved work) never appear in the state.

**D4 — Factorized retrieval + listwise rerank (the brief's hybrid).**
- *Benefit:*
  - Cost flat in candidate count.
  - Frees positional budget, so skill descriptions can return in stage 1.
  - Scales to apps, files, windows and controls; replaces the 12 ms lexical shortlist.
- *Math:*
  - At K=8 the rerank pass is ≈110 tokens vs 372, ≈3.4× fewer FLOPs [I].
  - Stage 1 adds a state encode plus 50–255 dot products, and a second graph launch.
  - On GPU at batch 1 (launch-bound) the gain may be ~0; on CPU it could be ~2–3× [S until
    measured].
- *Complexity:* high (two models, two ONNX graphs, gating logic, parity).
- *Novelty:* low to moderate (cascades and retrieve-then-rerank are standard [A]).
- *Risk:*
  - Stage-1 recall caps accuracy. Zero-shot BGE top-3 on grounding is only 76.9% [M].
  - Abstention candidates ("clarify", "none") retrieve poorly by cosine and must always be
    included.
  - A "clearly separated → execute" path would bypass the calibrated stage, which is unsafe
    unless the risk gate still runs.
  - The reranker must be retrained on hard top-K sets (distribution shift).
- *Compute:* two to three runs.
- *Falsifier:* at 50 skills, reject if hybrid top-1 < v6 − 0.3 pp (paired), **or** CPU
  latency improves < 2×, **or** stage-1 recall@8 < 99.5%. At 100–255 candidates, reject if it
  does not beat lexical-shortlist + listwise on accuracy at equal latency.

**D5 — Late-interaction split (DeFormer/PreTTR-style [A]) plus an exact state-KV cache.**
- *What:* in layers 1..k, options attend only to their own tokens (state-independent, so
  cacheable per option text). Layers k+1..12 run the full `block_bidir` branch. Cache the
  lower-layer option states and the state's K/V.
- *Benefit:* one model and one graph, graded compute; candidate cost scales by (12−k)/12.
  Keeps listwise interaction and abstention in the top layers.
- *Complexity:* moderate (mask schedule by layer, a cache in C#; ONNX needs per-layer inputs,
  which is the main engineering risk).
- *Novelty:* moderate in this setting, since no public System-One replication does it
  [I, from the research reports].
- *Risk:* accuracy loss at high k; the ONNX/DirectML export of split graphs.
- *Compute:* a k-sweep of ~5 short runs.
- *Falsifier:* if accuracy drops more than 0.5 pp before k reaches 6 (half the layers
  cached), caching is not worth its complexity.

**D6 — Hard-negative curriculum** (confusion-matrix siblings, embedding neighbours; CLM-style
staging).
- *Benefit:* sharper sibling discrimination (alarm vs reminder vs timer; snooze vs delete).
- *Complexity:* low.
- *Novelty:* low.
- *Risk:* many "confusions" are label noise or state-dependent (§1.6), so mining them teaches
  noise.
- *Compute:* matched-step comparison.
- *Falsifier:* at matched steps, hard negatives must beat the same number of generic rows on a
  frozen *confusion-pair panel* (paired). CLM's evidence is README-only.

**D7 — Reliability / selective-prediction layer.**
- *Benefit:* higher auto-execute coverage at a fixed error rate. Potentially large for
  grounding, where P(none) plus margin plus app familiarity carry signal a temperature
  cannot use.
- *Complexity:* low.
- *Novelty:* low.
- *Risk:* Kev found reliability heads *worse* than one well-fitted temperature.
- *Falsifier:* it must beat temperature scaling on AURC and coverage@≤1% error on a locked
  app-disjoint panel (paired bootstrap), with features fitted on a separate calibration
  split.

**D8 — Capacity control: BGE-base (110M).**
- *Benefit:* tells us whether capacity limits grounding or state use.
- *Complexity:* trivial.
- *Novelty:* none.
- *Compute:* ~3× v6.
- *Falsifier/diagnostic:* if BGE-base does not improve held-out-app grounding, capacity is not
  the bottleneck.

**D9 — Trajectory and personalisation post-training from app logs** (accepted/overridden
actions, clicked controls).
- *Benefit:* real distribution; long term.
- *Complexity:* high (logging, privacy, replay).
- *Risk:* forgetting. Kev's stacking failure is the warning; use joint retraining with replay.
- *Falsifier:* personal gains without a regression on the frozen benchmark. Defer until the
  app logs exist.

---

## 9. Recommended v7

**Choose D1 + D3 + D2 as the primary line, with D5 vs D4 as the compute track, and D7/D8 as
diagnostics.** This is not the brief's proposal in its proposed form. The evidence:

- The measured failures are grounding abstention and ranking, missing state, and desktop
  coverage. None is caused by the listwise architecture.
- The fixed 50-skill question is near its label-noise ceiling and already interactive on GPU.
- The hybrid's benefits (flat cost in candidate count, freed positional budget) are real for
  **dynamic, high-cardinality candidates**. It should be tested there.
- The DeFormer-style split is a single-model alternative that keeps listwise interaction and
  abstention, and it has not been compared.

**Primary hypothesis H7** [H]:
> Adding explicit device state to the state representation, and training grounding on an
> app-diverse corpus with shortcut-free abstention, raises:
> (i) state-conditioned minimal-pair accuracy from v6's marginal rate to ≥95%;
> (ii) held-out-app on-screen control selection from 48.7% to ≥75%, with control-question
> ECE ≤0.05;
> while real-command skill accuracy on the locked panel does not drop by more than 0.3 pp
> (paired bootstrap, 95%).

Falsified if (i) or (ii) misses its bar, or the guard fails.

**Compute hypothesis H7c** [H]:
> For candidate sets of 50–255, a candidate-cached design keeps ≥99.5% of full-listwise top-1
> (relative) while cutting CPU latency by ≥2× and making latency flat in candidate count.
> Designs: D4, the retrieve-then-rerank cascade, or D5, the lower-layer cache.

Falsified per design by the criteria in §8.

**v7-a also includes state-dependent risk** (added 2026-09-26, §14) [H]:
> Risk is trained and judged per (action, observable state) using the written rubric
> (`jevletbench-v1.md` §7), not per skill.
> On P5, v7 must show 0 unsafe autonomous executions on R3 items and a false-safe rate on R2+R3
> no higher than v6's, without over-blocking R0 items by more than 2 pp.
> Matched minimal pairs, where the same action changes tier with observable state, form a required
> training family. Unobservable factors are scored at their worst plausible value.

**BGE-base (D8) is deferred.** It is used only if, after the data and state fixes, the 33M model
still clearly saturates below the H7 targets, so that capacity does not compensate for defective
data.

Operational prerequisites (not research):
1. Export v6 to ONNX and run golden parity.
2. Add a model/catalogue version check.
3. Commit the C# app.
4. Fix the §1.10 defects.
5. Correct the public model card's training time. It is outward-facing, so confirm first.

---

## 10. Experimental plan (in order)

| # | Step | Output | Gate |
|---|---|---|---|
| 0 | Ops: v6 ONNX export and parity; version check; commit the app; fix `colab.py` progress restore and per-run timing; fix the evaluator groupings (per-question grounding, soft-target risk metrics, `is_ood` rename) | Clean v6 baseline in the app | **Done locally 2026-09-26** (68 C# and 141 Python tests pass); commit/push and the HF card correction await owner approval |
| 1 | **Build and freeze JevletBench-v1** (`jevletbench-v1.md`): owner-authored and public items, labelled before any v7 data is written; hash manifest; register release criteria | Frozen panels | Gate 3; design awaiting approval |
| 2 | **Baselines on JevletBench-v1** (dev + calibration splits only; locked read once): v6, v4, lexical, zero-shot BGE cosine, BGE-embedding logistic regression; energy via `nvidia-smi` power sampling where possible | Baseline table | Gate 6 |
| 3 | Measurement check: can v4 vs v6 be separated per panel? | CI widths | Resize panels if not |
| 4 | **E-state (D3):** v6 recipe + device-state lines + minimal pairs; 1 seed screening, 3 seeds for the finalist | Minimal-pair and guard results | H7(i) |
| 5 | **E-ground (D2):** app-diverse grounding + abstention fix; ablate each part (diversity only / abstention only / both) | Held-out-app curve | H7(ii) |
| 6 | Diagnostics: D8 (BGE-base) and D7 (reliability layer vs temperature) on grounding | Is capacity or calibration the limit? | – |
| 7 | **E-compute A/B/C/D** on the candidate-scaling panel (4/8/16/50/100/255 candidates): **A** v6 packed; **B** pure contrastive BGE-small bi-encoder (InfoNCE, duplicate-positive masking, unclamped scale); **C** B top-K → A-style reranker retrained on hard top-K sets, K ∈ {4, 8, 16}; **D** DeFormer split, k ∈ {3, 6, 9}. Measure top-1, recall@K, ECE/NLL/Brier, abstention, order flips (≥1,000 items × 4 orders), CPU and GPU latency (cold/warm/cached), VRAM, FLOPs | Frontier plot | H7c |
| 8 | Combine the winners into the v7 candidate; register criteria; **one** locked read | v7 decision | – |
| 9 | Cross-system comparison (§11.4) | Paper table | – |

Autoresearch v2 applies from step 4 onward. A contained mutable module
(`jevlet/research/v7_module.py`: heads, loss, sampling, curriculum, top-K gate) is the only
thing the agent may edit. Data, splits, evaluation code (hash-checked) and budget are
immutable. Keep/revert is decided on **dev** with a paired bootstrap, never on test (CLM's
flaw). Finalists get 3 seeds. The vault and the locked panels never reach the agent.
Negative results are logged in full (Kev's practice).

The Jevlet-S causal prefix/KV-reuse experiment (brief Phase 14) is **deprioritised**
(§1.4). If it runs, it should be measured in the regime where it matters: long states
(≥512 tokens) and ≥16 questions, reporting FLOPs and latency.

---

## 11. Benchmark protocol: JevletBench-v1

The full proposed design is in **`research/jevletbench-v1.md`** (revision 2 of 2026-09-27, awaiting
owner approval; not frozen). It covers:
- panels and sizes;
- provenance and the authorship rule;
- the dev/calibration/locked/external split policy;
- contamination rules;
- the statistical protocol;
- release criteria;
- the safety rubric;
- the state-observability spec;
- the manifest format with hashes.

Summary of what changed from the first draft of this section:
- **Authorship rule** (§14): the agent that writes training templates writes no natural-language
  benchmark items. Desktop and state panels come from the owner's phrasing and local palette usage;
  other natural-language panels come from source-separated public data. Claude builds only
  structure and programmatic panels.
- **Sources:** HWU64, STOP and multilingual TOP are excluded as likely sharing origins with
  MASSIVE/TOPv2. The first draft listed some of them as candidates.
- **Sizes** were reset to what one owner can author. v1.0 holds the panels that test H7 and H7c;
  v1.1 adds the rest under the same rules. Precision is stated per panel and re-checked by a pilot
  on dev before the freeze.
- **Risk** is scored with a five-factor rubric per (action, observable state), with tiers R0–R3.
  Unobservable factors are scored at their worst plausible value.

### 11.4 Identical cross-system benchmark (the brief's Phase 17)

- **Format:** a portable JSONL: `{state, questions:[{text, kind, options:[{name,
  description}]}], gold, acceptable, flags}`. Built from the external split of panels 1–4,
  7, 10 and 13.
- **Systems:**
  - Jevlet v6, contrastive Jevlet (B), hybrid (C), DeFormer (D).
  - Kev-0.8B: local A2000 4 GB is plausible for inference at short lengths [I]; otherwise
    Colab.
  - CLM-8B: Colab A100 only; ~16 GB encoder [I].
  - Jev: only if API access reopens (signups were paused 22 Sep per press [X]).
- **Fairness:** report zero-shot and in-domain-trained results separately. The fair
  architecture comparison is to fine-tune CLM heads and a Kev-0.8B LoRA on the **same**
  Jevlet training data. Otherwise Jevlet's in-domain advantage is confounded with
  architecture.
- **Metrics:** accuracy, Brier, ECE, NLL, cold/warm/cached latency, candidate scaling, peak
  memory, active params, FLOPs per decision, throughput, and energy where measurable.

---

## 12. Paper / research framing

**Is it preprint-worthy now?** No [I]. The results are in-distribution, the benchmarks are
self-authored, there are no baselines, and the grounding and OOD reporting is flawed.

**What would make it one:**

1. **Framing.** *"Minimum activated compute for calibrated local action selection."*
   - Jevlet's regime (short per-keystroke state, static candidates, local, safety-gated) is
     the inverse of Jev and Kev's long-shared-state regime.
   - The paper would show which structural choices buy the most accuracy, calibration and
     safety per FLOP: listwise interaction, candidate caching (retrieve-rerank vs
     lower-layer cache), and explicit state.
2. **Contributions.** Contributions (a) and (b) are the core; (c) is secondary.
   - (a) A compute–accuracy–calibration–unsafe-autonomy frontier on a frozen public benchmark
     (JevletBench-v1 external split) covering a 33M packed listwise model, its factorized and
     late-interaction variants, Kev-0.8B and CLM-8B, trained-on-same-data and zero-shot.
   - (b) State-conditioned minimal pairs: small models exploit explicit device state, and
     text-only routing cannot.
   - (c) An analysis of abstention shortcuts in UI grounding: the confident-"none" failure and
     its fix.
3. **Required honesty.**
   - Strong simple baselines: lexical, zero-shot cosine, SetFit-style fine-tuned bi-encoder,
     BGE+LR.
   - Kev-style locked reads.
   - Reporting where Jevlet loses: knowledge, long context, unseen apps.
4. **Prior art to position against** [A; verify before citing]:
   - Cascades and routing: FrugalGPT, RouteLLM.
   - Late interaction and decomposition: ColBERT, DeFormer, PreTTR.
   - Selective classification: Geifman & El-Yaniv.
   - Calibration: Guo et al. 2017.
   - Few-shot sentence-transformer classification: SetFit.
   - Plus Jev, Kev and CLM.
5. **Timing risk.** The space moved from zero to about ten open replications in ten days
   (§2). A narrow, well-measured claim beats a broad one.

The size ratios (~24× vs Kev-0.8B, ~239× vs the CLM-8B encoder) are parameter ratios only.
The paper should use FLOPs per decision and measured latency and energy.

---

## 13. Roadmap

| Phase | Work | Exit criterion |
|---|---|---|
| Now (no training) | Step 0 ops fixes; v6 in the app; JevletBench-v1 authored and frozen; baselines recorded | Gates 3 and 6 closed |
| v7-a | E-state and E-ground (a few Colab runs); diagnostics D7/D8 | H7 accepted or rejected |
| v7-b | Compute track A/B/C/D on the cardinality panel; C# candidate cache for the winner | H7c accepted or rejected per design |
| v7-rc | Combined candidate; one locked read; app release | Registered criteria met |
| Paper | Cross-system benchmark; frontier; write-up | External split results with CIs |
| Later | Trajectory/personalisation from app logs with replay (D9) | No regression on frozen panels |

### Residual risks in this audit

- **Error audit:** the 40-error sample was judged by one annotator (me). The gold-wrong,
  ambiguous and model-wrong split is indicative, with roughly ±15-point uncertainty per
  bucket at n=40.
- **Vault overlap:** assumed similar to dev, not measured (to keep the vault untouched).
- **Unverified FLOPs, latency and memory:** figures for the hybrid, Kev and CLM are
  inferences, not measurements.
- **Unchecked prior-session claims:** the v4 4.2e-5 ONNX deviation and the int8 trade-off were
  not re-derived (v6's deviation was: 2.72e-5).
- **External claims:** these come from research agents that read code and docs. They
  re-checked key claims against raw files but ran no Kev, CLM or Jev models. CLM's Notion
  blog was unreadable.
- **Grounding test:** the held-out grounding set is synthetic (2 apps). Real-app behaviour
  may be better or worse; only panel 4 can say.

---

## 14. Decisions (owner, 2026-09-26)

These are part of the plan. They override anything earlier in this document that conflicts with
them.

**1. Step 0 authorised.**
- Authorised: the v6 ONNX export and parity check, the strict model/catalogue check, the
  reporting and resume fixes, and the JevletBench-v1 design.
- **Public actions need explicit owner approval of the exact diff first:**
  - pushing the C# app and the other commits to the public GitHub repo;
  - editing the Hugging Face model card.

**2. Model-card training time.** The card will read:
> ≈5 h total A100 training (estimated): 3.33 h measured for steps 8,550–25,000 after a resume;
> first-session duration not recorded.

The measured part stays exact and the estimate is labelled as one.

**3. State-dependent risk is part of v7-a**, under these constraints:
- Only state the production app observes reliably at run time (the tier A/B fields in
  `jevletbench-v1.md` §8). Privileged or convenience fields are never invented.
- A written rubric (irreversibility, external effect, scope, privacy/security, authority → tiers
  R0–R3; `jevletbench-v1.md` §7). Unobservable factors are scored at their worst plausible value.
- Risk is generated and trained at the **(action, observable state)** level, never as a per-skill
  constant.
- Matched minimal pairs, where the same action changes risk with observable state, are a required
  training family.
- The eventual locked safety test (P5) stays separate from all training examples and their seeds.

**4. Benchmark authorship rule** (binding; `jevletbench-v1.md` §2.1):
- The same agent does not author both natural-language benchmark items and the training templates
  that will be optimised against them. Claude authors v7 templates, so it authors no
  natural-language benchmark items and assigns no labels on locked items.
- Desktop-command and state-conditioned panels come primarily from the owner's phrasing and, where
  practical, real palette usage. Raw personal material stays local; only hashes or sanitised
  material are published.
- Other natural-language panels come from public sources separated from all training sources.
- Claude may design structure, schemas, validation and sampling, plus programmatic panels
  (permutations, candidate-set mutations, corruption transforms, cardinality sweeps, latency).
- Every panel records its authorship and source provenance in the manifest.

**5. No v7 data and no v7 training** until the JevletBench-v1 manifest and release criteria are
approved, committed and frozen.

**6. BGE-base (D8) is deferred** until the data and state fixes are in, and only if the 33M model
still clearly saturates below target.

**7. Primary line confirmed:** D1 + D3 + D2 (measurement, observable state, app-diverse grounding
with better abstention data). Then the compute track: A current listwise / B pure contrastive /
C contrastive top-K → listwise / D lower-layer cached listwise, at K = 4, 8, 16, 50, 100 and 255.

### Decisions and actions (owner, 2026-09-27)

**1. Published.**
- The three step-0 commits are pushed to GitHub (`72717c5`).
- The Hugging Face model card is corrected on `main` (commit `0d8a6ee5`):
  - per-question grounding (Evaluation A, 3,000 examples, and Evaluation B, 30,000 examples,
    labelled separately), with the false-abstention share (94.6% of target-on-screen errors);
  - the soft-target risk note (D) and the in-sample calibration note (E);
  - training time with its measured and estimated parts distinguished;
  - loss-log coverage;
  - the training-progression relabel and two figures (loss at the logged steps only; one-run
    progression with learning-rate factors).
- The `v6` tag is unchanged. Weights, results and progress files are identical between `v6` and
  `main`.

**2. Wording rule for grounding claims.** Teams and Spotify are held out **from the grounding
training split** only. They appear by name elsewhere in training (the command generators' app
lists), so no stronger isolation is claimed.

**3. Authorship, amended.**
- Natural-language panels come from a mixture of genuine opt-in palette usage, some commands
  written by the owner, multiple independent human authors where practical, and source-separated
  public data.
- The owner does not hand-write thousands of items (a single author's style is its own narrow
  distribution).
- Provenance is recorded per item or cluster.
- Central paper claims need a sufficiently large **public, reproducible** external evaluation.
  Private hash-only panels support product validation only.

**4. Unknown state is not maximum risk.**
- *Action risk given known state* is kept separate from *state sufficiency for autonomous
  execution*.
- When a safety-relevant variable is unobservable, the policy asks for confirmation or abstains,
  without labelling the action itself R3.

**5. Statistics, amended.**
- Skill guards are paired non-inferiority tests with registered margins (−0.3 pp vault skill,
  −1.0 pp desktop commands).
- Safety reports one-sided upper confidence bounds, with enough R3 clusters for the bound to mean
  something (≈300 for ≈1%).
- The abstention guard is replaced by correct abstention, false abstention, coverage at ≤1/2/5%
  error, and AURC.
- The option-order test is sized to bound the ≤1% flip target.
- End-to-end palette latency (median and p95, with a stage breakdown) is recorded next to the
  model-pass ceilings.

**6. Grounding panel.**
- At least 20 apps, with no small set of apps dominating, app-disjoint from training where
  possible.
- Explicit strata: target present, target absent, near-duplicate controls, unfamiliar vocabulary,
  cross-app lexical similarity, and controls whose meaning depends on state.
- "Unfamiliar wording → none" must not remain exploitable.

**7. Collection over authoring.** Programmatic transforms multiply independent base items; the base
item or cluster stays the statistical unit.

**8. Unchanged:**
- The fixed 50-skill listwise architecture stays as the v6 baseline.
- D4 and D5 are both kept and tested where cardinality makes caching useful (apps, files,
  windows, controls, 50–255 candidates).
- BGE-base stays a diagnostic, run only after the data and state interventions are tested on
  BGE-small.
- No v7 data or training until JevletBench-v1 is approved and frozen.
