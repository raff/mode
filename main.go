package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/go-audio/audio"
	"github.com/go-audio/transforms"
	"github.com/go-audio/wav"
	"github.com/gordonklaus/portaudio"
	"github.com/jroimartin/gocui"
)

const (
	minToneFreq = 300
	maxToneFreq = 2000
)

// Morse code mapping
var morseCode = map[string]string{
	// letters
	".-": "A", "-...": "B", "-.-.": "C", "-..": "D",
	".": "E", "..-.": "F", "--.": "G", "....": "H",
	"..": "I", ".---": "J", "-.-": "K", ".-..": "L",
	"--": "M", "-.": "N", "---": "O", ".--.": "P",
	"--.-": "Q", ".-.": "R", "...": "S", "-": "T",
	"..-": "U", "...-": "V", ".--": "W", "-..-": "X",
	"-.--": "Y", "--..": "Z",

	// digits
	".----": "1", "..---": "2",
	"...--": "3", "....-": "4", ".....": "5", "-....": "6",
	"--...": "7", "---..": "8", "----.": "9", "-----": "0",

	// punctuations
	".-..-.": "\"", "...-..-": "$", ".----.": "'" /*"-.--.": "[",*/, "-.--.-": "]",
	/*".-.-.": "+",*/ "--..--": ",", "-....-": "-", ".-.-.-": ".", ".-.-.-.": ".",
	"-..-.": "/", "---...": ":", "-.-.-.": ";" /*"-...-": "=",*/, "..--..": "?",
	".--.-.": "@", "..--.-": "_", "-.-.--": "!", "---.": "!",

	// prosigns
	".-.-.": "<AR/+>", ".-...": "<AS>", "-...-.-": "<BK>", "...-.-": "<SK>", "-...-": "<BT/=>", "-.--.": "<KN/[>",
}

// Biquad filter struct, for bandpass filtering
type Biquad struct {
	a, b [3]float64
	x, y [2]float64
}

func NewBandpass(sampleRate, center, bandwidth float64) *Biquad {
	Q := center / bandwidth
	omega := 2 * math.Pi * center / sampleRate
	alpha := math.Sin(omega) / (2 * Q)
	b0 := alpha
	b1 := 0.0
	b2 := -alpha
	a0 := 1 + alpha
	a1 := -2 * math.Cos(omega)
	a2 := 1 - alpha
	return &Biquad{
		b: [3]float64{b0 / a0, b1 / a0, b2 / a0},
		a: [3]float64{1, a1 / a0, a2 / a0},
	}
}

func (f *Biquad) Filter(x float64) float64 {
	y := f.b[0]*x + f.b[1]*f.x[0] + f.b[2]*f.x[1] - f.a[1]*f.y[0] - f.a[2]*f.y[1]
	f.x[1] = f.x[0]
	f.x[0] = x
	f.y[1] = f.y[0]
	f.y[0] = y
	return y
}

// Applies a bandpass filter to a FloatBuffer's Data (modifies in place)
func DenoiseMorse(buf *audio.FloatBuffer, low, high float64) {
	center := (low + high) / 2
	bw := high - low
	bpf := NewBandpass(float64(buf.Format.SampleRate), center, bw)
	for i, s := range buf.Data {
		buf.Data[i] = bpf.Filter(s)
	}
}

// FFT performs a Fast Fourier Transform using Cooley-Tukey algorithm
func FFT(x []complex128) []complex128 {
	n := len(x)
	if n <= 1 {
		return x
	}

	// Ensure power of 2
	if n&(n-1) != 0 {
		// Pad to next power of 2
		nextPow2 := 1
		for nextPow2 < n {
			nextPow2 <<= 1
		}
		padded := make([]complex128, nextPow2)
		copy(padded, x)
		x = padded
		n = nextPow2
	}

	// Base case
	if n == 1 {
		return x
	}

	// Divide
	even := make([]complex128, n/2)
	odd := make([]complex128, n/2)
	for i := 0; i < n/2; i++ {
		even[i] = x[2*i]
		odd[i] = x[2*i+1]
	}

	// Conquer
	even = FFT(even)
	odd = FFT(odd)

	// Combine
	result := make([]complex128, n)
	for k := 0; k < n/2; k++ {
		t := cmplx.Exp(complex(0, -2*math.Pi*float64(k)/float64(n))) * odd[k]
		result[k] = even[k] + t
		result[k+n/2] = even[k] - t
	}

	return result
}

// DetectDominantFrequency finds the dominant frequency in the signal using FFT
func DetectDominantFrequency(data []float64, sampleRate int, minFreq, maxFreq float64) (float64, float64) {
	if len(data) <= 2 {
		return 0.0, 0.0
	}

	// Use a reasonable window size (power of 2)
	windowSize := 8192
	if len(data) < windowSize {
		windowSize = 1
		for windowSize < len(data) {
			windowSize <<= 1
		}
		windowSize >>= 1
	}

	// Take the first windowSize samples
	window := data
	if len(data) > windowSize {
		window = data[:windowSize]
	}

	// Convert to complex for FFT
	complexData := make([]complex128, windowSize)
	hammingSum := 0.0

	for i := 0; i < len(window); i++ {
		// Apply Hamming window to reduce spectral leakage
		hammingWindow := 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(len(window)-1))
		hammingSum += hammingWindow
		complexData[i] = complex(window[i]*hammingWindow, 0)
	}

	// Perform FFT
	fftResult := FFT(complexData)

	// Calculate magnitude spectrum
	magnitudes := make([]float64, len(fftResult)/2)
	for i := 0; i < len(magnitudes); i++ {
		magnitudes[i] = cmplx.Abs(fftResult[i])
	}

	// Find peak frequency within the specified range
	freqResolution := float64(sampleRate) / float64(len(fftResult))
	minBin := int(minFreq / freqResolution)
	maxBin := int(maxFreq / freqResolution)

	if minBin < 0 {
		minBin = 0
	}
	if maxBin >= len(magnitudes) {
		maxBin = len(magnitudes) - 1
	}

	peakBin := minBin
	peakMagnitude := magnitudes[minBin]

	for i := minBin; i <= maxBin; i++ {
		if magnitudes[i] > peakMagnitude {
			peakMagnitude = magnitudes[i]
			peakBin = i
		}
	}

	// Calculate the actual frequency
	peakFrequency := float64(peakBin) * freqResolution

	// Parabolic interpolation for more accurate frequency estimation
	if peakBin > 0 && peakBin < len(magnitudes)-1 {
		alpha := magnitudes[peakBin-1]
		beta := magnitudes[peakBin]
		gamma := magnitudes[peakBin+1]

		p := 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
		peakFrequency = (float64(peakBin) + p) * freqResolution
	}

	return peakFrequency, 2 * peakMagnitude / hammingSum
}

type ToneType int

const (
	Silence ToneType = iota
	Sound
)

// ToneSegment represents a detected tone with start and end times
type ToneSegment struct {
	Type      ToneType // sound, silence or short silence
	StartTime float64  // Start time in seconds
	EndTime   float64  // End time in seconds
	StartIdx  int      // Start sample index
	EndIdx    int      // End sample index
	Duration  float64  // Duration in seconds
	Magnitude float64
}

func (t ToneSegment) String() string {
	tt := map[ToneType]string{Silence: "S", Sound: "T"}[t.Type]
	return fmt.Sprintf("<%v %v (%.1f)>", tt, int(t.Duration*1000), t.Magnitude)
}

func String(t *ToneSegment) string {
	if t == nil {
		return "<nil>"
	}

	return (*t).String()
}

// ComputeEnvelope calculates the amplitude envelope of the signal
func ComputeEnvelope(data []float64, windowSize int) []float64 {
	envelope := make([]float64, len(data))

	for i := range data {
		sum := 0.0
		count := 0

		// Calculate RMS over a sliding window
		start := i - windowSize/2
		end := i + windowSize/2

		if start < 0 {
			start = 0
		}
		if end > len(data) {
			end = len(data)
		}

		for j := start; j < end; j++ {
			sum += data[j] * data[j]
			count++
		}

		if count > 0 {
			envelope[i] = math.Sqrt(sum / float64(count))
		}
	}

	return envelope
}

// SmoothSignal applies a simple moving average filter
func SmoothSignal(data []float64, windowSize int) []float64 {
	smoothed := make([]float64, len(data))

	for i := range data {
		sum := 0.0
		count := 0

		start := i - windowSize/2
		end := i + windowSize/2

		if start < 0 {
			start = 0
		}
		if end > len(data) {
			end = len(data)
		}

		for j := start; j < end; j++ {
			sum += data[j]
			count++
		}

		if count > 0 {
			smoothed[i] = sum / float64(count)
		}
	}

	return smoothed
}

// DetectMorseTones finds the beginning and end of each Morse tone in the signal
func DetectMorseTones(buf *audio.FloatBuffer, wpm int, thresholdRatio, centerFreq, bandwidth, noiseGate float64, debug bool) []ToneSegment {
	sampleRate := buf.Format.SampleRate

	// Calculate envelope with window size appropriate for the sample rate
	// For Morse code, typical dot duration is 50-100ms, so we use ~10ms window
	windowSize := sampleRate / 100 // 10ms window
	if windowSize < 10 {
		windowSize = 10
	}

	envelope := ComputeEnvelope(buf.Data, windowSize)

	// Smooth the envelope to reduce noise
	smoothWindowSize := sampleRate / 200 // 5ms smoothing
	if smoothWindowSize < 5 {
		smoothWindowSize = 5
	}
	envelope = SmoothSignal(envelope, smoothWindowSize)

	// Calculate adaptive threshold
	// Find the maximum and minimum envelope values
	maxEnv := 0.0
	minEnv := envelope[0]
	for _, v := range envelope {
		if v > maxEnv {
			maxEnv = v
		}
		if v < minEnv {
			minEnv = v
		}
	}

	// Apply Noise Gate (Squelch)
	// If the maximum signal is below the noise gate, ignore the entire buffer
	if maxEnv < noiseGate {
		return nil
	}

	// Threshold is a percentage between min and max
	threshold := minEnv + (maxEnv-minEnv)*thresholdRatio
	//fmt.Printf("Envelope range: %.6f to %.6f, threshold: %.6f\n", minEnv, maxEnv, threshold)

	// Detect tone segments using hysteresis thresholding
	var segments []ToneSegment
	inTone := false
	var startIdx, endIdx int

	// Use hysteresis: higher threshold to start, lower to continue
	startThreshold := threshold
	endThreshold := threshold * 0.8
	minDuration := ditTime(wpm) / 5 // minimum duration to consider a valid tone. Under this is likely noise

	addTone := func(start, end int, ttype ToneType, mag, minDur float64) bool {
		if start < 0 || end <= start {
			log.Println("addtone: invalid tone segment:", start, end, ttype)
			return false
		}

		// Calculate times
		startTime := float64(start) / float64(sampleRate)
		endTime := float64(end) / float64(sampleRate)
		duration := endTime - startTime

		// Filter out very short segments (likely noise)
		if duration > minDur {
			segments = append(segments, ToneSegment{
				Type:      ttype,
				StartTime: startTime,
				EndTime:   endTime,
				StartIdx:  start,
				EndIdx:    end,
				Duration:  duration,
				Magnitude: mag,
			})

			return true
		}

		return false
	}

	last := len(envelope) - 1

	for i, v := range envelope {
		vm := v * 100 / maxEnv

		if !inTone && v > startThreshold {
			// Tone begins
			added := false
			if i == 0 {
				added = true
			} else {
				added = addTone(endIdx, i, Silence, vm, minDuration) // add previous silence
			}

			if added {
				inTone = true
				startIdx = i
				endIdx = -1
			} else {
				// Silence was too short (dip in tone) -> ignore it and continue as tone
				if len(segments) > 0 {
					last := segments[len(segments)-1]
					if last.Type == Sound {
						// Remove the last segment (Sound) so we can extend it
						segments = segments[:len(segments)-1]
						inTone = true
						startIdx = last.StartIdx
						endIdx = -1
					}
				}
			}
		} else if inTone && v < endThreshold {
			// Tone ends
			if addTone(startIdx, i, Sound, vm, minDuration) {
				inTone = false
				endIdx = i
				startIdx = -1
			} else {
				// Sound was too short (spike in silence) -> ignore it and continue as silence
				if len(segments) > 0 {
					last := segments[len(segments)-1]
					if last.Type == Silence {
						// Remove the last segment (Silence) so we can extend it
						segments = segments[:len(segments)-1]
						inTone = false
						endIdx = last.StartIdx
						startIdx = -1
					}
				}
			}
		}
	}

	if endIdx > 0 {
		lastd := float64(last-endIdx) / float64(sampleRate)

		if lastd <= minDuration { // just add to last sameple
			l := len(segments) - 1
			segments[l].EndIdx = last
			segments[l].EndTime += lastd
			segments[l].Duration += lastd
		} else if !inTone && endIdx < last {
			addTone(endIdx, last, Silence, 100.0, 0) // add last silence
		}
	} else if inTone && startIdx >= 0 {
		addTone(startIdx, last, Sound, 100.0, 0) // add last tone
	}

	return segments
}

type MorseDecoder struct {
	code string
	wpm  int

	// Exponential moving average timing values (in milliseconds)
	ditTime int // average dit duration
	dahTime int // average dah duration
	mSpace  int // space between morse signals (dit/dah)
	chSpace int // space between characters
	wSpace  int // space between words

	tmag, smag float64

	cm []byte
}

func NewMorseDecoder(wpm int) *MorseDecoder {
	dTime := ditTime(wpm)

	return &MorseDecoder{
		wpm:     wpm,
		ditTime: int(dTime * 1000),
		dahTime: int(dTime * 3000),
		mSpace:  int(dTime * 1000),
		chSpace: int(dTime * 3000),
		wSpace:  int(dTime * 7000),
	}
}

func (d *MorseDecoder) Flush() string {
	if d.code == "" {
		return ""
	}
	result := d.deCode(d.code)
	d.code = ""
	return result
}

func ditTime(wpm int) float64 {
	if wpm <= 0 {
		wpm = 25 // default WPM
	}
	return 1.2 / float64(wpm) // in seconds
}

func ditTimeMs(wpm int) int {
	if wpm <= 0 {
		wpm = 25 // default WPM
	}
	return 1200 / wpm
}

func (d *MorseDecoder) deCode(code string) string {
	if code == "" {
		return ""
	}

	if val, ok := morseCode[code]; ok {
		return val
	}

	if false {
		var res string

		for i, c := range code {
			res += fmt.Sprintf("[%c %v]", c, d.cm[i])
		}

		return res
	} else {
		return "(" + code + ")"
	}

}

var minSpeed int = ditTimeMs(5)
var maxSpeed int = ditTimeMs(50)

func (d *MorseDecoder) Decode(segments []ToneSegment) string {
	var text string

	/*
		dtime := min(ditTimeMs(d.wpm), d.ditTime)
		if dtime > minSpeed {
			dtime = minSpeed
		} else if dtime < maxSpeed {
			dtime = maxSpeed
		}

		stime := min(ditTimeMs(d.wpm), d.chSpace)
		if stime > minSpeed {
			stime = minSpeed
		} else if stime < maxSpeed {
			stime = maxSpeed
		}
	*/

	dtime := ditTimeMs(d.wpm)
	stime := ditTimeMs(d.wpm)

	for _, seg := range segments {
		durMs := int(seg.Duration * 1000) // Convert to milliseconds

		switch seg.Type {
		case Silence:
			if durMs > 4*stime { // 7
				d.wSpace = (d.wSpace + durMs) / 2

				text += d.deCode(d.code) + " "
				d.code = ""

				d.cm = d.cm[:0]
			} else if durMs > 2*stime { // 3
				d.chSpace = (d.chSpace + durMs) / 2

				text += d.deCode(d.code)
				d.code = ""

				d.cm = d.cm[:0]
			} else if durMs > stime/2 { // 1
				d.mSpace = (d.mSpace + durMs) / 2
			}

			d.smag = (d.smag + seg.Magnitude) / 2
		case Sound:
			if durMs > 2*dtime { // 3
				// Dah - exponential moving average
				d.dahTime = (d.dahTime + durMs) / 2

				d.code += "-"
				d.cm = append(d.cm, byte(seg.Magnitude))
			} else if durMs > dtime*2/3 { // 1
				// Dit - exponential moving average
				d.ditTime = (d.ditTime + durMs) / 2

				d.code += "."
				d.cm = append(d.cm, byte(seg.Magnitude))
			} else {
				//d.code += "?"
			}

			d.tmag = (d.tmag + seg.Magnitude) / 2
		}
	}

	return text
}

func listAudioDevices() ([]string, error) {
	devices, err := portaudio.Devices()
	if err != nil {
		return nil, err
	}

	var list []string

	for _, d := range devices {
		v := d.Name

		if d.MaxInputChannels > 0 {
			v += fmt.Sprintf(" (in:%v)", d.MaxInputChannels)
		}
		if d.MaxOutputChannels > 0 {
			v += fmt.Sprintf(" (out:%v)", d.MaxOutputChannels)
		}

		list = append(list, v)
	}

	return list, nil
}

type AudioWriter struct {
	Stream       *portaudio.Stream
	StreamBuffer audio.Float32Buffer
	Volume       float32
	mute         bool
}

func NewAudioWriter(dev string, sampleRate, ssize int) (*AudioWriter, error) {
	devices, err := portaudio.Devices()
	if err != nil {
		return nil, err
	}

	var info *portaudio.DeviceInfo

	i, err := strconv.Atoi(dev)
	if err == nil && i > 0 && i <= len(devices) {
		info = devices[i-1]
	}

	if info == nil {
		for _, d := range devices {
			if info == nil && strings.HasPrefix(d.Name, dev) {
				info = d
			}

		}
	}

	if info == nil {
		return nil, fmt.Errorf("device not found: %s", dev)
	}

	var stream *portaudio.Stream

	const numChannels = 1

	p := portaudio.HighLatencyParameters(nil, info)
	p.Input.Channels = 0
	p.Output.Channels = numChannels
	p.SampleRate = float64(sampleRate)
	p.FramesPerBuffer = sampleRate / ssize

	buf32 := audio.Float32Buffer{Format: &audio.Format{NumChannels: numChannels, SampleRate: sampleRate}, Data: make([]float32, p.FramesPerBuffer)}

	stream, err = portaudio.OpenStream(p, buf32.Data)
	if err != nil {
		return nil, fmt.Errorf("open output: %w", err)
	}

	if err := stream.Start(); err != nil {
		return nil, fmt.Errorf("start output: %w", err)
	}

	return &AudioWriter{
		Stream:       stream,
		StreamBuffer: buf32,
		Volume:       1.0,
	}, nil
}

func (w *AudioWriter) Close() {
	if w.Stream != nil {
		w.Stream.Stop()
	}
}

func (w *AudioWriter) Mute(m bool) {
	if w.mute = m; w.mute {
		for i := range w.StreamBuffer.Data {
			w.StreamBuffer.Data[i] = 0
		}
	}
}

func (w *AudioWriter) Write(b *audio.FloatBuffer) error {
	if w.mute {
		return nil
	}

	buf32 := b.AsFloat32Buffer()

	if w.Volume != 1.0 {
		for i := 0; i < len(buf32.Data); i++ {
			buf32.Data[i] *= w.Volume
		}
	}

	copy(w.StreamBuffer.Data, buf32.Data)
	return w.Stream.Write()
}

type AudioReader struct {
	Stream       *portaudio.Stream
	StreamBuffer audio.Float32Buffer

	WavDecoder *wav.Decoder
	WavBuffer  audio.IntBuffer

	SampleRate int
	Channels   int
	SampleSize int
}

func FromWaveFile(r io.ReadSeeker, ssize int) (*AudioReader, error) {
	decoder := wav.NewDecoder(r)
	if !decoder.IsValidFile() {
		return nil, fmt.Errorf("invalid WAV file")
	}

	return &AudioReader{
		WavDecoder: decoder,
		WavBuffer:  audio.IntBuffer{Format: decoder.Format(), Data: make([]int, decoder.Format().SampleRate/ssize)},
		SampleRate: decoder.Format().SampleRate,
		Channels:   decoder.Format().NumChannels,
		SampleSize: ssize,
	}, nil
}

func FromAudioStream(dev string, ssize int) (*AudioReader, error) {
	devices, err := portaudio.Devices()
	if err != nil {
		return nil, err
	}

	var info *portaudio.DeviceInfo

	i, err := strconv.Atoi(dev)
	if err == nil && i > 0 && i <= len(devices) {
		info = devices[i-1]
	}

	if info == nil {
		for _, d := range devices {
			if info == nil && strings.HasPrefix(d.Name, dev) {
				info = d
				break
			}
		}
	}

	if info == nil {
		return nil, fmt.Errorf("device not found: %s", dev)
	}

	const numChannels = 1
	const sampleRate = 44100

	p := portaudio.HighLatencyParameters(info, nil)
	p.Input.Channels = numChannels
	p.Output.Channels = 0
	p.SampleRate = sampleRate
	p.FramesPerBuffer = sampleRate / ssize

	buf32 := audio.Float32Buffer{Format: &audio.Format{NumChannels: numChannels, SampleRate: sampleRate}, Data: make([]float32, p.FramesPerBuffer)}

	stream, err := portaudio.OpenStream(p, buf32.Data)
	if err != nil {
		return nil, fmt.Errorf("open input: %w", err)
	}

	if err := stream.Start(); err != nil {
		return nil, fmt.Errorf("start input: %w", err)
	}

	return &AudioReader{
		Stream:       stream,
		StreamBuffer: buf32,
		SampleRate:   sampleRate,
		Channels:     numChannels,
		SampleSize:   ssize,
	}, nil
}

func (r *AudioReader) Close() {
	if r.Stream != nil {
		r.Stream.Stop()
	}
	if r.WavDecoder != nil {
		// nothing to close
	}
}

func (r *AudioReader) Read() (*audio.FloatBuffer, int, error) {
	if r.Stream != nil {
		// Read from PortAudio stream
		r.StreamBuffer.Data = r.StreamBuffer.Data[:]

		if err := r.Stream.Read(); err != nil {
			return nil, 0, err
		}

		// Convert to FloatBuffer
		fb := r.StreamBuffer.AsFloatBuffer()
		transforms.NormalizeMax(fb)
		return fb, len(r.StreamBuffer.Data), nil
	}

	if r.WavDecoder != nil {
		// Read from WAV file
		r.WavBuffer.Data = r.WavBuffer.Data[:]

		n, err := r.WavDecoder.PCMBuffer(&r.WavBuffer)
		if n == 0 {
			return nil, 0, nil
		}
		if err != nil {
			return nil, 0, err
		}

		// Adjust buffer size if needed
		// (it appears that PCMBuffer may return less samples at the end of the file)
		r.WavBuffer.Data = r.WavBuffer.Data[:n]

		// Convert to FloatBuffer
		fb := r.WavBuffer.AsFloatBuffer()
		transforms.NormalizeMax(fb)

		return fb, n, nil
	}

	return nil, 0, fmt.Errorf("no audio source available")
}

type App struct {
	gui   *gocui.Gui
	vinfo *gocui.View
	vmain *gocui.View
	vcmd  *gocui.View

	startTime time.Time

	reader    *AudioReader
	player    *AudioWriter
	mode      *MorseDecoder
	threshold int
	sep       bool
	bandwidth float64 // frequency bandwidth for bandpass filter
	noiseGate float64 // minimum amplitude to consider as signal

	duration int
	tone     int
	mag      float64

	mute bool
}

func (app *App) Layout(g *gocui.Gui) (err error) {
	maxX, maxY := g.Size()

	app.vinfo, err = g.SetView("info", 0, 0, maxX-1, 2)
	if err != nil {
		if err != gocui.ErrUnknownView {
			return err
		}

		app.vinfo.Title = "MoDe - Morse Decoder"
	}

	app.vmain, err = g.SetView("main", 0, 3, maxX-1, maxY-5)
	if err != nil {
		if err != gocui.ErrUnknownView {
			return err
		}

		app.vmain.Title = "Decoded"
		app.vmain.Wrap = true
		app.vmain.Autoscroll = true
	}

	app.vcmd, err = g.SetView("cmdline", 0, maxY-4, maxX-1, maxY-1)
	if err != nil {
		if err != gocui.ErrUnknownView {
			return err
		}

		app.vcmd.Title = "Available commands"
		fmt.Fprintf(app.vcmd, "^C/^Q: quit  b: -bandwidth  w: -wpm  n: -noise t: -threshold")

		if app.player != nil {
			fmt.Fprintf(app.vcmd, "  v: -volume")
		}

		fmt.Fprintf(app.vcmd, "  s: toggle separator\n")
		fmt.Fprintf(app.vcmd, "c: clear     B: +bandwidth  W: +wpm  N: +noise T: +threshold")

		if app.player != nil {
			fmt.Fprintf(app.vcmd, "  V: +volume  m: toggle audio/mute")
		}
	}

	d := time.Since(app.startTime)
	app.vinfo.Clear()
	app.vinfo.SetOrigin(0, 0)
	fmt.Fprintf(app.vinfo,
		"WPM: %2d (%2d) dit: %-2dms sp: %-2d/%-3dms  NG: %3.1f Thr: %2d%%  Bw: %3d Freq: %3d Level: %3d (T: %3d S: %3d)   %8v",
		app.mode.wpm,
		1200/app.mode.ditTime,
		app.mode.ditTime,
		app.mode.mSpace,
		app.mode.wSpace,
		app.noiseGate,
		app.threshold,
		int(app.bandwidth),
		app.tone,
		int(app.mag*1000),
		int(app.mode.tmag), // *1000),
		int(app.mode.smag), // *1000),
		d.Truncate(time.Second).String(),
	)

	if app.player != nil {
		fmt.Fprintf(app.vinfo, " vol: %d", int(app.player.Volume*10))
	}

	return nil
}

func (app *App) SetKeyBinding() error {

	//
	// quit application: CtrlC / CtrlQ
	//

	quit := func(g *gocui.Gui, v *gocui.View) error {
		return gocui.ErrQuit
	}

	if err := app.gui.SetKeybinding("", gocui.KeyCtrlC, gocui.ModNone, quit); err != nil {
		return err
	}
	if err := app.gui.SetKeybinding("", gocui.KeyCtrlQ, gocui.ModNone, quit); err != nil {
		return err
	}

	//
	// clear screen: c
	//

	clearscreen := func(g *gocui.Gui, v *gocui.View) error {
		app.vmain.Clear()
		return nil
	}

	if err := app.gui.SetKeybinding("", 'c', gocui.ModNone, clearscreen); err != nil {
		return err
	}

	//
	// toggle separator: s
	//

	toggleSep := func(g *gocui.Gui, v *gocui.View) error {
		app.sep = !app.sep
		return nil
	}

	if err := app.gui.SetKeybinding("", 's', gocui.ModNone, toggleSep); err != nil {
		return err
	}

	//
	// bandwidth up/down: B / b
	//

	bandwidthUp := func(g *gocui.Gui, v *gocui.View) error {
		if app.bandwidth < 500 {
			app.bandwidth += 50
		}

		return nil
	}

	bandwidthDown := func(g *gocui.Gui, v *gocui.View) error {
		if app.bandwidth > 50 {
			app.bandwidth -= 50
		}

		return nil
	}

	if err := app.gui.SetKeybinding("", 'B', gocui.ModNone, bandwidthUp); err != nil {
		return err
	}

	if err := app.gui.SetKeybinding("", 'b', gocui.ModNone, bandwidthDown); err != nil {
		return err
	}

	//
	// wpm up/down: W / w
	//

	wpmUp := func(g *gocui.Gui, v *gocui.View) error {
		if app.mode.wpm < 50 {
			app.mode.wpm++
		}

		return nil
	}

	wpmDown := func(g *gocui.Gui, v *gocui.View) error {
		if app.mode.wpm > 1 {
			app.mode.wpm--
		}

		return nil
	}

	if err := app.gui.SetKeybinding("", 'W', gocui.ModNone, wpmUp); err != nil {
		return err
	}

	if err := app.gui.SetKeybinding("", 'w', gocui.ModNone, wpmDown); err != nil {
		return err
	}

	//
	// noise gate up/down: N / n
	//

	noiseUp := func(g *gocui.Gui, v *gocui.View) error {
		if app.noiseGate < 1 {
			app.noiseGate += 0.1
		}

		return nil
	}

	noiseDown := func(g *gocui.Gui, v *gocui.View) error {
		if app.noiseGate > 0.1 {
			app.noiseGate -= 0.1
		}

		return nil
	}

	if err := app.gui.SetKeybinding("", 'N', gocui.ModNone, noiseUp); err != nil {
		return err
	}

	if err := app.gui.SetKeybinding("", 'n', gocui.ModNone, noiseDown); err != nil {
		return err
	}

	//
	// threshold up/down: T / t
	//

	thresholdUp := func(g *gocui.Gui, v *gocui.View) error {
		if app.threshold < 100 {
			app.threshold += 5
		}

		return nil
	}

	thresholdDown := func(g *gocui.Gui, v *gocui.View) error {
		if app.threshold > 5 {
			app.threshold -= 5
		}

		return nil
	}

	if err := app.gui.SetKeybinding("", 'T', gocui.ModNone, thresholdUp); err != nil {
		return err
	}

	if err := app.gui.SetKeybinding("", 't', gocui.ModNone, thresholdDown); err != nil {
		return err
	}

	if app.player != nil {

		//
		// toggle mute: m
		//

		toggleMute := func(g *gocui.Gui, v *gocui.View) error {
			app.mute = !app.mute
			app.player.Mute(app.mute)
			return nil
		}

		if err := app.gui.SetKeybinding("", 'm', gocui.ModNone, toggleMute); err != nil {
			return err
		}

		//
		// volume up/down: V / v
		//

		volumeUp := func(g *gocui.Gui, v *gocui.View) error {
			if app.player.Volume < 2.0 {
				app.player.Volume += 0.1
			}

			return nil
		}

		volumeDown := func(g *gocui.Gui, v *gocui.View) error {
			if app.player.Volume > 0.0 {
				app.player.Volume -= 0.1
			}

			return nil
		}

		if err := app.gui.SetKeybinding("", 'V', gocui.ModNone, volumeUp); err != nil {
			return err
		}

		if err := app.gui.SetKeybinding("", 'v', gocui.ModNone, volumeDown); err != nil {
			return err
		}

	}

	return nil
}

func (app *App) Print(s string) {
	app.gui.Update(func(g *gocui.Gui) error {
		fmt.Fprint(app.vmain, s)
		return nil
	})
}

func (app *App) MainLoop() {
	var toneSegments []ToneSegment
	var prevTone *ToneSegment

	// Threshold ratio: 0.3 means 30% above the minimum envelope value
	// Adjust this value if detection is too sensitive (lower it) or misses tones (raise it)

	for {
		thresholdRatio := float64(app.threshold) / 100.0

		floatBuf, n, err := app.reader.Read()
		if err != nil {
			app.Print("\n\n" + err.Error() + "\n\n")
			time.Sleep(10 * time.Second)
			return
		}

		if n == 0 {
			break
		}

		// Convert to mono (in place)
		transforms.MonoDownmix(floatBuf)

		// denoise ?
		//DenoiseMorse(floatBuf, minToneFreq, maxToneFreq)

		d := float64(len(floatBuf.Data)) / float64(app.reader.SampleRate)
		app.duration += int(d * 1000)

		// Automatically detect the Morse code tone frequency
		centerFreq, magnitude := DetectDominantFrequency(floatBuf.Data, floatBuf.Format.SampleRate, minToneFreq, maxToneFreq)

		app.tone = int(centerFreq)
		app.mag = magnitude

		// Calculate low and high cutoff frequencies
		lowCutoff := centerFreq - app.bandwidth/2
		highCutoff := centerFreq + app.bandwidth/2

		// Ensure reasonable bounds
		if lowCutoff < minToneFreq {
			lowCutoff = minToneFreq
		}
		if highCutoff > maxToneFreq {
			highCutoff = maxToneFreq
		}

		// Apply bandpass filter centered on detected frequency
		DenoiseMorse(floatBuf, lowCutoff, highCutoff)

		if app.player != nil {
			app.player.Write(floatBuf)
		}

		// Detect Morse code tone segments (beginning and end of each tone)
		toneSegments = DetectMorseTones(floatBuf, app.mode.wpm, thresholdRatio, centerFreq, app.bandwidth, app.noiseGate, false)

		/* if *debug {
			fmt.Println("segments:", toneSegments, "prev:", String(prevTone))
		} */

		if prevTone != nil {
			if len(toneSegments) == 0 { // can this still happen ?
				continue
			}

			if prevTone.Type == toneSegments[0].Type {
				// merge
				toneSegments[0].StartIdx = prevTone.StartIdx
				toneSegments[0].Duration += prevTone.Duration
			} else {
				toneSegments = append([]ToneSegment{*prevTone}, toneSegments...)
			}

			prevTone = nil
		}

		n = len(toneSegments)
		if n == 0 {
			continue
		}

		n--
		prevTone = &toneSegments[n]
		if n == 0 {
			continue
		}

		toneSegments = toneSegments[:n]
		if app.sep {
			app.Print("_")
		}

		if text := app.mode.Decode(toneSegments); text != "" {
			app.Print(text)
		}
	}

	if prevTone != nil {
		if app.sep {
			app.Print("_")
		}

		if text := app.mode.Decode([]ToneSegment{*prevTone}); text != "" {
			app.Print(text)
		}
	}

	// Flush any remaining code
	if text := app.mode.Flush(); text != "" {
		if app.sep {
			app.Print("_")
		}

		app.Print(text)
	}

	app.Print("\n\nDone\n\n")
}

func main() {
	ssize := flag.Int("buffer", 300, "buffer size (in ms)")
	//debug := flag.Bool("debug", false, "debug messages")
	wpm := flag.Int("wpm", 20, "words per minute (for timing)")
	dev := flag.String("device", "", "input audio device (for live decoding)")
	out := flag.String("play", "", "output audio device (for monitoring)")
	bandwidth := flag.Float64("bandwidth", 300, "bandwidth for bandpass filter (in Hz)")
	threshold := flag.Int("threshold", 50, "Threshold ratio (percentage)")
	noiseGate := flag.Float64("noisegate", 0.01, "Noise gate (squelch) level (0.0-1.0)")
	flag.Parse()

	if *threshold < 1 {
		*threshold = 1
	}
	if *threshold > 100 {
		*threshold = 100
	}

	if *dev != "" || *out != "" || flag.NArg() == 0 {
		// Initialize PortAudio
		err := portaudio.Initialize()
		if err != nil {
			log.Fatalf("Failed to initialize PortAudio: %v", err)
		}
		defer portaudio.Terminate()
	}

	if *dev == "" && flag.NArg() == 0 {
		fmt.Println()
		fmt.Printf("Usage: %v [options] [filename]\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
		fmt.Println()

		l, err := listAudioDevices()
		if err != nil {
			log.Fatal(err)
		}

		fmt.Println("Available audio devices")
		for i, d := range l {
			fmt.Println("", i+1, d)
		}

		din, _ := portaudio.DefaultInputDevice()
		dout, _ := portaudio.DefaultOutputDevice()

		fmt.Println()
		if din != nil {
			fmt.Println("Default input device:", din.Name)
		}
		if dout != nil {
			fmt.Println("Default output device:", dout.Name)
		}
		return
	}

	*ssize = 1000 / *ssize
	if *ssize <= 0 {
		*ssize = 1
	}

	var reader *AudioReader
	var player *AudioWriter
	var err error

	if *dev != "" {
		reader, err = FromAudioStream(*dev, *ssize)
		if err != nil {
			log.Fatal(err)
		}
	} else if flag.NArg() >= 1 {
		inputFile := flag.Arg(0)

		f, err := os.Open(inputFile)
		if err != nil {
			log.Fatal(err)
		}

		defer f.Close()

		reader, err = FromWaveFile(f, 1) // *ssize)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		log.Fatal("no input source specified")
	}

	if *out != "" {
		player, err = NewAudioWriter(*out, reader.SampleRate, *ssize)
		if err != nil {
			log.Fatal(err)
		}
	}

	g, err := gocui.NewGui(gocui.OutputNormal)
	if err != nil {
		log.Panicln(err)
	}
	defer g.Close()

	app := App{
		gui:       g,
		startTime: time.Now(),
		bandwidth: *bandwidth,
		threshold: *threshold,
		noiseGate: *noiseGate,
		reader:    reader,
		player:    player,
		mode:      NewMorseDecoder(*wpm),
	}

	g.SetManagerFunc(app.Layout)

	app.SetKeyBinding()

	go app.MainLoop()

	if err := g.MainLoop(); err != nil && err != gocui.ErrQuit {
		log.Panicln(err)
	}
}
