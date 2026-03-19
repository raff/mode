# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
make gmode        # build TUI app (tmode binary)
make tmode        # build TUI app (tmode binary)
make gui          # run GUI app directly (go run gmain.go decoder.go)
make tui          # run TUI app directly (go run main.go decoder.go)
make text         # run TUI app without interactive UI (go run main.go decoder.go -noui)
make app          # build macOS MoDe.app bundle (requires fyne tool)
make clean        # remove built binaries and MoDe.app

go build -o gmode gmain.go decoder.go   # GUI (Fyne)
go build -o tmode main.go decoder.go    # TUI (gocui)
```

There are no tests in this project.

## Architecture

**MoDe** (Morse Decoder) decodes Morse code from live audio (via PortAudio) or WAV files. The codebase has three files:

### `decoder.go` — shared core logic
The heart of the application. Contains:
- **Signal processing**: `ComputeEnvelope`, `SmoothSignal`, `MedianFilter`, FFT/spectrum analysis (`FFT`, `ComputeSpectrum`, `DetectDominantFrequency`, `Spectrogram`)
- **Audio filters**: `Biquad`/`BiquadCascade` for bandpass filtering (`Denoise`), `NewBoost`/`AudioPeakFilter` for peak EQ
- **Tone detection**: `DetectMorseTones` — segments audio into Sound/Silence `ToneSegment` values using adaptive hysteresis thresholding
- **Morse decoding**: `MorseDecoder` — maps tone segments to dit/dah, then looks up `morseCode` map. Uses adaptive timing (exponential moving averages) and optional 2-means clustering for dit/dah classification
- **Audio I/O**: `AudioReader` (PortAudio stream or WAV file via `go-audio/wav`), `AudioWriter` (PortAudio output)
- **`DecoderApp`**: The shared application struct holding all decoder state. Both UIs embed or instantiate this. Its `MainLoop()` runs in a goroutine — reads audio chunks, runs spectrum analysis, applies filter, detects tones, and calls `AddText`/`SetStatus`/`Update` callbacks provided by the UI layer

### `gmain.go` — GUI frontend (Fyne)
Uses `fyne.io/fyne/v2` for a cross-platform GUI. Embeds UbuntuMono fonts from `assets/fonts/`. Defines custom widgets (`NumericStepper`, `TextLog`, `BorderedContainer`) and a `CompactTheme`. Instantiates `DecoderApp` and calls `go modeApp.MainLoop()`.

### `main.go` — TUI frontend (gocui)
Uses `github.com/jroimartin/gocui` for a terminal UI. Has a `//go:build nobuild` tag — it is **not compiled by default** and must be explicitly built. Provides key bindings for live parameter adjustment (bandwidth, WPM, noise gate, threshold, filter, volume). Instantiates `DecoderApp` and calls `go app.MainLoop()`.

### Key design points
- `DecoderApp.MainLoop()` is UI-agnostic; UIs wire in callbacks (`AddText`, `SetStatus`, `Update`)
- Frequency detection is auto-tracked per audio chunk with exponential smoothing
- Morse timing adapts in real time via moving averages on observed dit/dah durations
- The `main.go` TUI build tag (`nobuild`) means `go build .` only builds the GUI. Use explicit file lists to build the TUI
- The TUI's `-noui` flag disables the interactive interface and writes decoded text to stdout — useful for testing: `go run main.go decoder.go -noui <file.wav>` will decode a WAV file containing a CW transmission and print the decoded words
- The macOS app bundle requires the `fyne` CLI tool (`go install fyne.io/fyne/v2/cmd/fyne@latest`)

## Design System
Always read `DESIGN.md` before making any visual or UI decisions.
All font choices, colors, spacing, and aesthetic direction are defined there.
Do not deviate without explicit user approval.
In QA mode, flag any code that doesn't match `DESIGN.md`.
