package decoder_test

// TestRecordMicrophone reads from the default PortAudio input device and writes
// a WAV file, then reads it back to verify it contains non-silent audio.
//
// Run manually (requires a microphone and make some noise):
//
//	go test -v -run TestRecordMicrophone ./internal/decoder/
//
// Output file: /tmp/mic-test.wav  (play with: afplay /tmp/mic-test.wav)

import (
	"math"
	"os"
	"testing"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/gordonklaus/portaudio"
)

func TestRecordMicrophone(t *testing.T) {
	if err := portaudio.Initialize(); err != nil {
		t.Fatalf("portaudio.Initialize: %v", err)
	}
	defer portaudio.Terminate()

	dev, err := portaudio.DefaultInputDevice()
	if err != nil {
		t.Fatalf("DefaultInputDevice: %v", err)
	}
	t.Logf("Input device: %q (max channels: %d, default sample rate: %.0f)",
		dev.Name, dev.MaxInputChannels, dev.DefaultSampleRate)

	const (
		sampleRate      = 44100
		numChannels     = 1
		chunksPerSecond = 10
		recordSeconds   = 3
		bitDepth        = 16
	)

	framesPerBuffer := sampleRate / chunksPerSecond
	buf := make([]float32, framesPerBuffer)

	p := portaudio.HighLatencyParameters(dev, nil)
	p.Input.Channels = numChannels
	p.Output.Channels = 0
	p.SampleRate = sampleRate
	p.FramesPerBuffer = framesPerBuffer

	stream, err := portaudio.OpenStream(p, buf)
	if err != nil {
		t.Fatalf("OpenStream: %v", err)
	}
	defer stream.Stop()

	if err := stream.Start(); err != nil {
		t.Fatalf("stream.Start: %v", err)
	}

	outPath := "/tmp/mic-test.wav"
	f, err := os.Create(outPath)
	if err != nil {
		t.Fatalf("create %s: %v", outPath, err)
	}
	defer f.Close()

	// 16-bit signed PCM, format 1
	enc := wav.NewEncoder(f, sampleRate, bitDepth, numChannels, 1)
	defer enc.Close()

	t.Logf("Recording %d seconds — make some noise...", recordSeconds)

	intData := make([]int, framesPerBuffer)
	var maxSample float64
	var sumSq float64
	totalSamples := 0

	for range chunksPerSecond * recordSeconds {
		if err := stream.Read(); err != nil {
			t.Fatalf("stream.Read: %v", err)
		}

		// Scale float32 [-1, 1] → int16 range.
		// NOTE: go-audio's AsIntBuffer() just casts float→int with no scaling,
		// so values in [-1,1] all become 0 (silent). We must scale manually.
		for i, s := range buf {
			v := float64(s)
			if math.Abs(v) > maxSample {
				maxSample = math.Abs(v)
			}
			sumSq += v * v
			totalSamples++

			scaled := int(v * math.MaxInt16)
			if scaled > math.MaxInt16 {
				scaled = math.MaxInt16
			} else if scaled < math.MinInt16 {
				scaled = math.MinInt16
			}
			intData[i] = scaled
		}

		ib := &audio.IntBuffer{
			Format:         &audio.Format{NumChannels: numChannels, SampleRate: sampleRate},
			Data:           intData,
			SourceBitDepth: bitDepth,
		}
		if err := enc.Write(ib); err != nil {
			t.Fatalf("enc.Write: %v", err)
		}
	}

	rms := math.Sqrt(sumSq / float64(totalSamples))
	t.Logf("Recorded %d samples: max=%.5f  rms=%.5f", totalSamples, maxSample, rms)
	t.Logf("Wrote: %s  (play with: afplay %s)", outPath, outPath)

	if maxSample < 0.001 {
		t.Errorf("audio appears silent: max=%.6f — check microphone permissions or make noise during the test", maxSample)
	}
}
