package decoder_test

import (
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-audio/audio"
	"github.com/go-audio/transforms"
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
		Wait:              false,
		Bandwidth:         300,
		Threshold:         50,
		MinSNR:            0.1,
		NoiseFloorPct:     20,
		Dither:            0,
		MinFreq:           300,
		MaxFreq:           2000,
		Reader:            reader,
		Mode:              decoder.NewMorseDecoder(20, 0, 0.75),
		Filter:            decoder.Denoise,
		SpectralPeakRatio: 3,
		AddText: func(s string) {
			sb.WriteString(s)
		},
	}

	app.MainLoop()

	return sb.String()
}

// TestNoiseGateSuppressesNoise verifies that a buffer of pure Gaussian noise
// produces no Sound segments after NormalizeMax — i.e., the minSNR gate fires.
// This is a regression test for the bug where the old AND condition
// (signalRef < noiseGate && snr < snrEps) never fired after NormalizeMax.
func TestNoiseGateSuppressesNoise(t *testing.T) {
	const sampleRate = 44100
	const durationSec = 1.0
	n := int(sampleRate * durationSec)

	rng := rand.New(rand.NewSource(42))
	data := make([]float64, n)
	for i := range data {
		data[i] = rng.NormFloat64() * 0.1
	}

	fb := &audio.FloatBuffer{
		Data:   data,
		Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
	}
	transforms.NormalizeMax(fb)

	// Pure Gaussian noise has SNR ≈ 0.018 after NormalizeMax; real CW chunks
	// have SNR ≥ 0.08.  A threshold of 0.1 sits cleanly in the gap.
	const minSNR = 0.1
	segments := decoder.DetectMorseTones(fb, 20, 0.5, 700, 300, minSNR, 20, 0)

	for _, seg := range segments {
		if seg.Type == decoder.Sound {
			t.Errorf("noise input produced Sound segment: %v", seg)
		}
	}
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
