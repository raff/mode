package decoder_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gordonklaus/portaudio"
	"github.com/raff/mode/internal/decoder"
)

// decodeWAV runs the decoder against a WAV file and returns the decoded text.
func decodeWAV(t *testing.T, wavPath string) string {
	t.Helper()

	f, err := os.Open(wavPath)
	if err != nil {
		t.Fatalf("open %s: %v", wavPath, err)
	}
	defer f.Close()

	reader, err := decoder.FromWaveFile(f, 1)
	if err != nil {
		t.Fatalf("FromWaveFile: %v", err)
	}

	var sb strings.Builder

	app := decoder.DecoderApp{
		Wait:          false,
		Bandwidth:     300,
		Threshold:     50,
		NoiseGate:     0.2,
		NoiseFloorPct: 20,
		Dither:        0,
		MinFreq:       300,
		MaxFreq:       2000,
		Reader:        reader,
		Mode:          decoder.NewMorseDecoder(20, 0, 0.75),
		Filter:        decoder.Denoise,
		SpectralPeakRatio: 3,
		AddText: func(s string) {
			sb.WriteString(s)
		},
	}

	app.MainLoop()

	return sb.String()
}

// TestWAVFixtures decodes every testdata/*.wav and compares against testdata/*.txt.
// The .txt file is the golden output: regenerate with `go run ./cmd/tmode -noui <file.wav>`.
func TestWAVFixtures(t *testing.T) {
	wavFiles, err := filepath.Glob("../../testdata/*.wav")
	if err != nil {
		t.Fatal(err)
	}
	if len(wavFiles) == 0 {
		t.Skip("no testdata/*.wav files found")
	}

	// PortAudio must be initialized even when using WAV files (imported by decoder).
	if err := portaudio.Initialize(); err != nil {
		t.Fatalf("portaudio.Initialize: %v", err)
	}
	defer portaudio.Terminate()

	for _, wavPath := range wavFiles {
		wavPath := wavPath
		name := strings.TrimSuffix(filepath.Base(wavPath), ".wav")

		t.Run(name, func(t *testing.T) {
			goldenPath := filepath.Join("../../testdata", name+".txt")
			golden, err := os.ReadFile(goldenPath)
			if err != nil {
				t.Skipf("no golden file %s — run `go run ./cmd/tmode -noui %s > testdata/%s.txt` to create it",
					goldenPath, wavPath, name)
			}

			got := decodeWAV(t, wavPath)
			want := strings.TrimRight(string(golden), "\n")

			if got != want {
				t.Errorf("decode(%s):\n  got:  %q\n  want: %q", name, got, want)
			}
		})
	}
}
