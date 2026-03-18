# TODOS

## P2 — Accuracy Benchmark Mode

**What:** Add a `-bench` flag that runs the decoder against `testdata/` WAV fixtures and reports character accuracy.

**Why:** Once the test suite exists, accuracy scoring lets you quantify the effect of filter and timing parameter changes — e.g., "changing squelch from 2 to 3 improved accuracy on noisy_cw.wav from 87% to 94%."

**Pros:** Turns subjective "sounds better" into a measurable metric. Makes parameter tuning scientific.

**Cons:** Requires good fixture coverage to be meaningful; edit-distance scoring needs careful implementation for morse (partial credit for near-misses).

**Context:** The `-noui` mode already decodes a WAV to stdout. The fixture test suite (added in the overhaul) provides the testdata/ infrastructure. `-bench` is a natural extension: instead of pass/fail, compute Levenshtein distance between decoder output and expected text.

**Effort:** M (human ~1 day) / S with CC (~20 min)

**Depends on:** WAV fixture test suite must exist first.

---

## P3 — Parameter Simplification: NoiseGate vs Threshold

**What:** Evaluate whether `NoiseGate` and `Threshold` can be unified into a single parameter inside `DetectMorseTones`.

**Why:** Both are amplitude-based silence gates. `NoiseGate` is a hard per-chunk gate (if overall signal `< noiseGate`, treat chunk as silence). `Threshold` is a relative per-sample crossing level (`noiseFloor + (signalRef − noiseFloor) × ratio`). Exposing both adds cognitive overhead for the operator.

**Pros:** Reduces control surface, simpler UI, fewer parameters to explain.

**Cons:** They do operate at different granularities — unifying them may lose a useful degree of freedom for marginal signals where chunk-level gating and sample-level crossing need independent tuning.

**Context:** `SpectralPeakRatio` (squelch) is NOT redundant with `Threshold` — they operate at different pipeline stages (frequency domain vs. time domain) and cannot replace each other. The overlap is specifically `NoiseGate` + `Threshold`. The test suite (WAV fixtures + accuracy benchmark) should be in place before attempting this so the effect can be measured.

**Effort:** S (human ~2h) / S with CC (~10 min)

**Depends on:** WAV fixture test suite and accuracy benchmark for before/after validation.

---

## P3 — Session Replay / Debug Mode

**What:** Given a WAV file and optionally a session log, replay the decode with `-verbose` output to understand where decoding diverged from expected.

**Why:** When a user reports "it decoded VVV as VVX", being able to replay the exact WAV with segment-level analysis makes root cause diagnosis fast.

**Pros:** Dramatically reduces time to diagnose timing/filter issues. Enables regression-driven development.

**Cons:** Requires the WAV to still be available (live audio sessions can't be replayed unless recorded).

**Context:** The `-verbose` flag (added in overhaul) logs segment durations and classifications to stderr. The session log records decoded text. A replay mode combines these: run `-noui -verbose -file foo.wav` and diff against the session log to find the divergence point.

**Effort:** M (human ~4h) / S with CC (~15 min)

**Depends on:** `-verbose` flag, session log — both added in overhaul.
