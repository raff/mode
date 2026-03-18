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

## P3 — Session Replay / Debug Mode

**What:** Given a WAV file and optionally a session log, replay the decode with `-verbose` output to understand where decoding diverged from expected.

**Why:** When a user reports "it decoded VVV as VVX", being able to replay the exact WAV with segment-level analysis makes root cause diagnosis fast.

**Pros:** Dramatically reduces time to diagnose timing/filter issues. Enables regression-driven development.

**Cons:** Requires the WAV to still be available (live audio sessions can't be replayed unless recorded).

**Context:** The `-verbose` flag (added in overhaul) logs segment durations and classifications to stderr. The session log records decoded text. A replay mode combines these: run `-noui -verbose -file foo.wav` and diff against the session log to find the divergence point.

**Effort:** M (human ~4h) / S with CC (~15 min)

**Depends on:** `-verbose` flag, session log — both added in overhaul.

---

## P3 — Refactor AudioReader as Interface

**What:** Split `AudioReader` into an interface with two concrete implementations: `WaveReader` (WAV file) and `StreamReader` (PortAudio live stream).

**Why:** `AudioReader` currently mixes two unrelated concerns in one struct — WAV file decoding and PortAudio streaming. Fields like `WavDecoder`/`WavBuffer` are nil for streams, and `Stream`/`StreamBuffer` are nil for files. An interface would make each implementation self-contained and easier to reason about.

**Pros:** Cleaner separation of concerns. Easier to add new sources (e.g., TCP stream, stdin). `Read()` dispatch becomes a simple vtable call instead of `if r.Stream != nil / if r.WavDecoder != nil`.

**Cons:** Requires updating all call sites that set fields directly on `AudioReader` (e.g., `RecordEncoder` would need to move to a wrapper or the interface). Moderate refactor.

**Context:** `AudioReader` is in `internal/decoder/decoder.go`. The interface would live there too. `FromWaveFile` and `FromAudioStream` would return the concrete types (or the interface directly).

**Effort:** M (human ~2h) / S with CC (~15 min)

**Depends on:** Nothing — can be done independently.

---

## P3 — Refactor Config to Handle CLI Flags

**What:** Move the flag-parsing + config-loading + merge logic into `internal/config` so both `gmode` and `tmode` mains call a single `config.LoadWithFlags()` instead of duplicating the `flag.Visit` + merge pattern.

**Why:** Both UIs contain identical boilerplate: declare flags, parse, load config, `flag.Visit` to find explicit flags, merge config defaults into flag values. This is ~25 lines duplicated verbatim.

**Pros:** Single source of truth for flag/config priority logic. Adding a new persisted setting requires one change instead of two.

**Cons:** The config package would need to import `flag` (acceptable) and know about all the flag names that map to config fields (mild coupling).

**Context:** The duplicated block lives at the top of `main()` in both `cmd/gmode/main.go` and `cmd/tmode/main.go`. A `config.Flags` struct + `config.RegisterFlags(fs *flag.FlagSet)` + `config.LoadWithFlags(fs *flag.FlagSet) Config` pattern would clean this up.

**Effort:** S (human ~1h) / S with CC (~10 min)

**Depends on:** Nothing — can be done independently.
