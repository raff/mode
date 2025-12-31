package main

import (
	"fmt"
	"io"
	"log"
	"math"
	"math/cmplx"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-audio/audio"
	"github.com/go-audio/transforms"
	"github.com/go-audio/wav"
	"github.com/gordonklaus/portaudio"
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

const nBands = 8 // number of frequency bands for spectrogram

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

// Boost designs a peaking EQ filter.
// gain: linear gain at peak (not dB!)
// fc: center frequency (Hz)
// bw: bandwidth (Hz) [default: fs/10]
// fs: sampling rate (Hz) [default: 1]
// Returns B (numerator) and A (denominator) coefficients.
func NewBoost(gain, fc, bw, fs float64) *Biquad {
	// Set defaults
	if fs == 0 {
		fs = 1
	}
	if bw == 0 {
		bw = fs / 10
	}

	Q := fs / bw
	wcT := 2 * math.Pi * fc / fs

	K := math.Tan(wcT / 2)
	V := gain

	// Numerator coefficients
	b0 := 1 + V*K/Q + K*K
	b1 := 2 * (K*K - 1)
	b2 := 1 - V*K/Q + K*K

	// Denominator coefficients
	a0 := 1 + K/Q + K*K
	a1 := 2 * (K*K - 1)
	a2 := 1 - K/Q + K*K

	// Normalize coefficients and return
	return &Biquad{
		b: [3]float64{b0 / a0, b1 / a0, b2 / a0},
		a: [3]float64{1, a1 / a0, a2 / a0},
	}
}

func (f *Biquad) Filter(x float64) float64 {
	y := f.b[0]*x + f.b[1]*f.x[0] + f.b[2]*f.x[1] - f.a[1]*f.y[0] - f.a[2]*f.y[1]
	f.x[1], f.x[0] = f.x[0], x
	f.y[1], f.y[0] = f.y[0], y
	return y
}

// Applies a bandpass filter to a FloatBuffer's Data (modifies in place)
func Denoise(buf *audio.FloatBuffer, low, high float64) {
	center := (low + high) / 2
	bw := high - low
	bpf := NewBandpass(float64(buf.Format.SampleRate), center, bw)
	for i, s := range buf.Data {
		buf.Data[i] = bpf.Filter(s)
	}
}

type AudioFilter func(buf *audio.FloatBuffer, low, high float64)

// Applies a boost filter to a FloatBuffer's Data (modifies in place)
func BoostSignal(buf *audio.FloatBuffer, gain, center, bandwidth float64) {
	boost := NewBoost(gain, center, bandwidth, float64(buf.Format.SampleRate))
	for i, s := range buf.Data {
		buf.Data[i] = boost.Filter(s)
	}
}

// apply peak boost filter (3db -> 1.413, 6db -> 2.0)
func AudioPeakFilter(gain float64) AudioFilter {
	return func(buf *audio.FloatBuffer, low, high float64) {
		BoostSignal(buf, gain, (low+high)/2, high-low)
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

// ComputeSpectrum performs FFT and returns the magnitude spectrum and the hamming window sum
func ComputeSpectrum(data []float64, sampleRate int) ([]float64, float64) {
	if len(data) <= 2 {
		return nil, 0.0
	}

	// Use a reasonable window size (power of 2)
	windowSize := 8192
	for len(data) < windowSize {
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
		magnitudes[i] = 2.0 / hammingSum * cmplx.Abs(fftResult[i])
	}

	return magnitudes, hammingSum
}

// DetectDominantFrequency finds the dominant frequency in the signal using the pre-computed spectrum
func DetectDominantFrequency(magnitudes []float64, hammingSum float64, sampleRate int, minFreq, maxFreq float64) (float64, float64) {
	if len(magnitudes) == 0 {
		return 0.0, 0.0
	}

	// Find peak frequency within the specified range
	// The spectrum size corresponds to half the window size used in ComputeSpectrum
	// We need to reconstruct the window size to calculate freqResolution
	windowSize := len(magnitudes) * 2
	freqResolution := float64(sampleRate) / float64(windowSize)

	minBin := int(minFreq / freqResolution)
	maxBin := int(maxFreq / freqResolution)

	if minBin < 0 {
		minBin = 0
	}
	if maxBin >= len(magnitudes) {
		maxBin = len(magnitudes) - 1
	}

	/*
		minBin := 0
		maxBin := len(magnitudes) - 1
	*/

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

	return peakFrequency, peakMagnitude
}

var levels = [8]rune{'\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588'}

// Spectrogram generates a  spectrogram representation from the magnitude spectrum
func Spectrogram(magnitudes []float64, hammingSum float64, sampleRate int, minFreq, maxFreq float64, graphic bool) (result [nBands]rune) {
	if len(magnitudes) == 0 {
		return result
	}

	windowSize := len(magnitudes) * 2
	freqResolution := float64(sampleRate) / float64(windowSize)

	minBin := int(minFreq / freqResolution)
	maxBin := int(maxFreq / freqResolution)

	if minBin < 0 {
		minBin = 0
	}
	if maxBin >= len(magnitudes) {
		maxBin = len(magnitudes) - 1
	}

	if maxBin <= minBin {
		return result
	}

	// Calculate band width in bins
	totalBins := maxBin - minBin + 1
	binsPerBand := float64(totalBins) / nBands

	for i := 0; i < nBands; i++ {
		startBin := minBin + int(float64(i)*binsPerBand)
		endBin := minBin + int(float64(i+1)*binsPerBand)
		if endBin > maxBin+1 {
			endBin = maxBin + 1
		}
		if startBin >= endBin {
			startBin = endBin - 1
		}

		sum := 0.0
		count := 0
		for j := startBin; j < endBin; j++ {
			if j >= 0 && j < len(magnitudes) {
				sum += magnitudes[j]
				count++
			}
		}

		var avg float64
		if count > 0 {
			avg = sum / float64(count)
		}

		// Normalize and map to 0-7
		// We use hammingSum to normalize, similar to DetectDominantFrequency
		// The 2.0 factor is from the windowing correction
		//normalized := 0.0
		//if hammingSum > 0 {
		//	normalized = 2 * avg / hammingSum
		//}

		// Map normalized magnitude to 0-7 levels
		// Assuming normalized is roughly 0.0 to 1.0 (or slightly more)
		// We can use a logarithmic scale or linear. Let's try linear first, scaled up.
		// Experimentally, values might be small.
		// Let's assume a dynamic range or just scale it.
		// For now, let's just multiply by a factor and clamp.
		// A value of 1.0 is a very loud signal (0dBFS sine wave).
		// Let's map 0.0-1.0 to 0-7.
		level := int(avg * 800)
		if level > 7 {
			level = 7
		}
		if graphic {
			result[i] = levels[level]
		} else {
			result[i] = rune('0' + level)
		}
	}

	return result
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
func DetectMorseTones(buf *audio.FloatBuffer, wpm int, thresholdRatio, centerFreq, bandwidth, noiseGate float64) []ToneSegment {
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
	} else if inTone && startIdx >= 0 && last > startIdx {
		addTone(startIdx, last, Sound, 100.0, 0) // add last tone
	}

	return segments
}

type MorseDecoder struct {
	code string
	wpm  int
	fwpm int
	dt   float64

	ditTime int // average dit duration
	dahTime int // average dah duration
	mSpace  int // space between morse signals (dit/dah)
	chSpace int // space between characters
	wSpace  int // space between words

	tmag, smag float64

	cm []byte
}

func NewMorseDecoder(wpm, fwpm int, dt float64) *MorseDecoder {
	if fwpm > wpm {
		fwpm = wpm
	}

	dtime := ditTimeMs(wpm)
	stime := spaceTimeMs(wpm, fwpm)

	return &MorseDecoder{
		wpm:     wpm,
		fwpm:    fwpm,
		dt:      dt,
		ditTime: dtime * 1,
		dahTime: dtime * 3,
		mSpace:  dtime * 1,
		chSpace: stime * 3,
		wSpace:  stime * 7,
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

func spaceTimeMs(wpm, fwpm int) int {
	if fwpm < 0 {
		fwpm += wpm
	}

	if fwpm > 0 && fwpm < wpm {
		// see https://www.arrl.org/files/file/Technology/x9004008.pdf
		ta := ((60.0 * float64(wpm)) - (37.2 * float64(fwpm))) / float64(fwpm*wpm)
		return int(ta * 1000 / 19)
	}

	return ditTimeMs(wpm)
}

func (d *MorseDecoder) getFwpm() int {
	if d.fwpm <= 0 {
		return d.fwpm + d.wpm
	}

	return d.fwpm
}

func (d *MorseDecoder) setFwpm(v int) {
	if v > d.wpm {
		d.fwpm = 0
	}

	d.fwpm = v - d.wpm
}

func (d *MorseDecoder) deCode(code string) string {
	if code == "" {
		return ""
	}

	if val, ok := morseCode[code]; ok {
		return val
	}

	if true {
		var imin = 0
		var tmin = 10000

		for i, t := range d.cm {
			if int(t) < tmin {
				tmin = int(t)
				imin = i
			}
		}

		if imin > 0 && imin < len(code)-1 {
			c1 := code[:imin]
			c2 := code[imin+1:]

			return d.deCode(c1) + d.deCode(c2)
		}

		return "(" + code + ")"
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
	stime := spaceTimeMs(d.wpm, d.fwpm)

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
			if durMs > int(float64(dtime*3)*d.dt) { // 3
				// Dah - exponential moving average
				d.dahTime = (d.dahTime + durMs) / 2

				d.code += "-"
				d.cm = append(d.cm, byte(durMs))
			} else if durMs > int(float64(dtime)*d.dt) { // 1
				// Dit - exponential moving average
				d.ditTime = (d.ditTime + durMs) / 2

				d.code += "."
				d.cm = append(d.cm, byte(durMs))
			} else {
				//d.code += "?"
			}

			d.tmag = (d.tmag + seg.Magnitude) / 2
		}
	}

	return text
}

type AudioType int

const (
	AudioInOut AudioType = iota
	AudioIn
	AudioOut
)

func ListAudioDevices(t AudioType) ([]string, error) {
	devices, err := portaudio.Devices()
	if err != nil {
		return nil, err
	}

	var list []string

	for _, d := range devices {
		v := d.Name

		switch t {
		case AudioInOut:
			if d.MaxInputChannels > 0 {
				v += fmt.Sprintf(" (in:%v)", d.MaxInputChannels)
			}
			if d.MaxOutputChannels > 0 {
				v += fmt.Sprintf(" (out:%v)", d.MaxOutputChannels)
			}

		case AudioIn:
			if d.MaxInputChannels == 0 { // output
				continue
			}

		case AudioOut:
			if d.MaxOutputChannels == 0 { // input
				continue
			}
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
	Id string // device name or filename

	Stream       *portaudio.Stream
	StreamBuffer audio.Float32Buffer

	WavDecoder *wav.Decoder
	WavBuffer  audio.IntBuffer

	SampleRate int
	Channels   int
	SampleSize int

	reading bool
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
		Id:           info.Name,
		Stream:       stream,
		StreamBuffer: buf32,
		SampleRate:   sampleRate,
		Channels:     numChannels,
		SampleSize:   ssize,
	}, nil
}

func (r *AudioReader) Close() {
	for r.reading {
		time.Sleep(100 * time.Millisecond)
	}

	if r.Stream != nil {
		r.Stream.Stop()
		r.Stream = nil
	}
	if r.WavDecoder != nil {
		// nothing to close
		r.WavDecoder = nil
	}
}

func (r *AudioReader) Rewind() error {
	if r.WavDecoder != nil {
		return r.WavDecoder.Rewind()
	}

	return nil
}

func (r *AudioReader) Read() (*audio.FloatBuffer, int, error) {
	r.reading = true
	defer func() {
		r.reading = false
	}()

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

type DecoderApp struct {
	MinFreq float64
	MaxFreq float64

	Wait bool // wait for more inputs
	mu   sync.Mutex

	Reader    *AudioReader
	Player    *AudioWriter
	Mode      *MorseDecoder
	Threshold int
	Bandwidth float64 // frequency bandwidth for bandpass filter
	NoiseGate float64 // minimum amplitude to consider as signal

	Duration int
	Tone     int
	Mag      float64

	Mute        bool
	Filter      AudioFilter
	Fname       string
	Spectrogram [nBands]rune

	AddText   func(s string)
	SetStatus func(s string)
	Update    func()
}

func (app *DecoderApp) SetReader(r *AudioReader) {
	app.mu.Lock()
	prev := app.Reader
	app.Reader = nil

	if prev != nil {
		app.mu.Unlock()
		prev.Close()
		prev = nil
		time.Sleep(500 * time.Millisecond)
		app.mu.Lock()
	}

	app.Reader = r
	app.mu.Unlock()

	app.Status("")
}

func (app *DecoderApp) GetReader() *AudioReader {
	app.mu.Lock()
	defer app.mu.Unlock()
	return app.Reader
}

func (app *DecoderApp) Print(s string) {
	if app.AddText != nil {
		app.AddText(s)
	}
}

func (app *DecoderApp) Status(s string) {
	log.Println("Status:", s)

	if app.SetStatus != nil {
		app.SetStatus(s)
	}
}

func (app *DecoderApp) Statusf(s string, args ...any) {
	app.Status(fmt.Sprintf(s, args...))
}

func (app *DecoderApp) MainLoop() {
	var toneSegments []ToneSegment
	var prevTone *ToneSegment

	// Threshold ratio: 0.3 means 30% above the minimum envelope value
	// Adjust this value if detection is too sensitive (lower it) or misses tones (raise it)

	for {
		reader := app.GetReader()
		if reader == nil {
			app.Status("Select audio input")
			time.Sleep(300 * time.Millisecond)
			continue
		}

		thresholdRatio := float64(app.Threshold) / 100.0

		floatBuf, n, err := reader.Read()
		if err != nil {
			app.Status("Error: " + err.Error())
			if app.Wait {
				app.SetReader(nil)
				continue
			} else {
				break
			}
		}

		if n == 0 {
			app.Status("No more data")
			if app.Wait {
				app.SetReader(nil)
				continue
			} else {
				break
			}
		}

		// Convert to mono (in place)
		transforms.MonoDownmix(floatBuf)

		d := float64(len(floatBuf.Data)) / float64(reader.SampleRate)
		app.Duration += int(d * 1000)

		// Automatically detect the Morse code tone frequency
		magnitudes, hammingSum := ComputeSpectrum(floatBuf.Data, floatBuf.Format.SampleRate)
		centerFreq, magnitude := DetectDominantFrequency(magnitudes, hammingSum, floatBuf.Format.SampleRate, app.MinFreq, app.MaxFreq)
		app.Spectrogram = Spectrogram(magnitudes, hammingSum, floatBuf.Format.SampleRate, app.MinFreq, app.MaxFreq, true)

		app.Tone = int(centerFreq)
		app.Mag = magnitude

		if centerFreq < app.MinFreq || centerFreq > app.MaxFreq {
			continue
		}

		// Calculate low and high cutoff frequencies
		lowCutoff := centerFreq - app.Bandwidth/2
		highCutoff := centerFreq + app.Bandwidth/2

		// Ensure reasonable bounds
		if lowCutoff < app.MinFreq {
			lowCutoff = app.MinFreq
		}
		if highCutoff > app.MaxFreq {
			highCutoff = app.MaxFreq
		}

		// Apply filter centered on detected frequency
		if app.Filter != nil {
			app.Filter(floatBuf, lowCutoff, highCutoff)
		}

		if app.Player != nil {
			app.Player.Write(floatBuf)
		}

		// Detect Morse code tone segments (beginning and end of each tone)
		toneSegments = DetectMorseTones(floatBuf, app.Mode.wpm, thresholdRatio, centerFreq, app.Bandwidth, app.NoiseGate)

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

		if text := app.Mode.Decode(toneSegments); text != "" {
			app.Print(text)
		}

		if app.Update != nil {
			app.Update()
		}
	}

	if prevTone != nil {
		if text := app.Mode.Decode([]ToneSegment{*prevTone}); text != "" {
			app.Print(text)
		}
	}

	// Flush any remaining code
	if text := app.Mode.Flush(); text != "" {
		app.Print(text)
	}

	app.Status("Done!")

	if app.Update != nil {
		app.Update()
	}
}
