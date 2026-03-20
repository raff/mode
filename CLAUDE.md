# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
make gmode        # build GUI app (gmode binary)
make tmode        # build TUI app (tmode binary)
make gui          # run GUI app directly
make tui          # run TUI app directly
make text         # run TUI app without interactive UI (-noui)
make app          # build macOS MoDe.app bundle (requires fyne tool)
make clean        # remove built binaries and MoDe.app

go build -o gmode ./cmd/gmode   # GUI (Fyne)
go build -o tmode ./cmd/tmode   # TUI (gocui)
```

```bash
go test ./internal/decoder/   # run decoder unit tests
```

Tests live in `internal/decoder/decoder_test.go`. WAV fixtures in `testdata/` are used by `TestWAVFixtures` — each `*.wav` is decoded and compared against a `*.txt` golden file. A companion `*.json` can override decoder params (`wpm`, `fwpm`, `st`, `minsnr`) per fixture. Regenerate a golden file with:
```bash
go run ./cmd/tmode -noui [flags] testdata/<name>.wav > testdata/<name>.txt
```

## Architecture

**MoDe** (Morse Decoder) decodes Morse code from live audio (via PortAudio) or WAV files.

```
cmd/gmode/      — GUI frontend (Fyne)
cmd/tmode/      — TUI frontend (gocui)
internal/decoder/ — shared core logic + tests
testdata/       — WAV fixtures and golden .txt files for decoder tests
```

### `internal/decoder/decoder.go` — shared core logic
The heart of the application. Contains:
- **Signal processing**: `ComputeEnvelope`, `SmoothSignal`, `MedianFilter`, FFT/spectrum analysis (`FFT`, `ComputeSpectrum`, `DetectDominantFrequency`, `Spectrogram`)
- **Audio filters**: `Biquad`/`BiquadCascade` for bandpass filtering (`Denoise`), `NewBoost`/`AudioPeakFilter` for peak EQ
- **Tone detection**: `DetectMorseTones` — segments audio into Sound/Silence `ToneSegment` values using adaptive hysteresis thresholding
- **Morse decoding**: `MorseDecoder` — maps tone segments to dit/dah, then looks up `morseCode` map. Uses adaptive timing (exponential moving averages) and optional 2-means clustering for dit/dah classification
- **Audio I/O**: `AudioReader` (PortAudio stream or WAV file via `go-audio/wav`), `AudioWriter` (PortAudio output)
- **`DecoderApp`**: The shared application struct holding all decoder state. Both UIs embed or instantiate this. Its `MainLoop()` runs in a goroutine — reads audio chunks, runs spectrum analysis, applies filter, detects tones, and calls `AddText`/`SetStatus`/`Update` callbacks provided by the UI layer

### `cmd/gmode/` — GUI frontend (Fyne)
Uses `fyne.io/fyne/v2` for a cross-platform GUI. Embeds UbuntuMono fonts from `assets/fonts/`. Defines custom widgets (`NumericStepper`, `TextLog`, `BorderedContainer`) and a `CompactTheme`. Instantiates `DecoderApp` and calls `go modeApp.MainLoop()`.

### `cmd/tmode/` — TUI frontend (gocui)
Uses `github.com/jroimartin/gocui` for a terminal UI. Provides key bindings for live parameter adjustment (bandwidth, WPM, noise gate, threshold, filter, volume). Instantiates `DecoderApp` and calls `go app.MainLoop()`.

### Key design points
- `DecoderApp.MainLoop()` is UI-agnostic; UIs wire in callbacks (`AddText`, `SetStatus`, `Update`)
- Frequency detection is auto-tracked per audio chunk with exponential smoothing
- Morse timing adapts in real time via moving averages on observed dit/dah durations
- The TUI's `-noui` flag disables the interactive interface and writes decoded text to stdout — useful for testing: `go run ./cmd/tmode -noui <file.wav>` will decode a WAV file containing a CW transmission and print the decoded words
- The macOS app bundle requires the `fyne` CLI tool (`go install fyne.io/fyne/v2/cmd/fyne@latest`)

## Design System
Always read `DESIGN.md` before making any visual or UI decisions.
All font choices, colors, spacing, and aesthetic direction are defined there.
Do not deviate without explicit user approval.
In QA mode, flag any code that doesn't match `DESIGN.md`.
