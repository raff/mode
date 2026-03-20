# mode
Morse Decoder

A morse decoder that can decode audio from a device (i.e. SDR or ham radio) or a file (wave files)

Usage: mode [options] [filename]

Notes:
- **Clean/synthetic audio** (e.g. computer-generated WAV files with no noise): use `-clean`. Real-world transmissions always have some background noise, which actually helps the envelope detector find tone boundaries. The `-clean` flag adds a light artificial noise floor (`-dither 0.001`) and disables the SNR gate (`-minsnr 0`) so that near-silent buffers are not suppressed. If the file also uses Farnsworth timing, set `-wpm` and `-fwpm` to match — e.g. `-wpm 25 -fwpm 10` for 25 WPM characters at 10 WPM text speed.
- **Noisy audio with Farnsworth timing**: if you know the recording speed, pass `-wpm` and `-fwpm` explicitly. Noise can compress individual dits below the default speed threshold (`-st 75`); lowering to `-st 50` accepts dits as short as 50% of the nominal dit duration, which helps in high-noise conditions.
- `-dither` can be used directly to fine-tune the artificial noise floor added to the envelope (overrides the value set by `-clean`). Use `0` to disable.

    -bandwidth float
    	bandwidth for bandpass filter (in Hz) (default 300)
    -buffer int
    	buffer size (in ms) (default 300)
    -clean
    	optimize for clean/synthetic audio: disables SNR gate and adds light envelope dither
    -device string
    	input audio device (for live decoding)
    -dither float
    	envelope dither amount (0 disables)
    -filter string
    	apply bandpass filter (bp), audio peak filter (apf), or no filter (none) (default "bp")
    -fwpm int
    	Farnsworth speed
    -list
    	list audio devices
    -maxfreq float
    	maximum frequency (in Hz) (default 2000)
    -minfreq float
    	minimum frequency (in Hz) (default 300)
    -minsnr float
    	minimum SNR (signal − noise floor) required to process a chunk; 0 disables (default 0.1)
    -noisepct float
    	percentile for noise floor estimation (1-80) (default 20)
    -noui
    	no user interface, write to stdout
    -play string
    	output audio device (for monitoring)
    -record
    	save incoming audio to a WAV file in the sessions folder
    -separator
    	output separator '_' between decoded segments
    -squelch int
    	squelch level (spectral peak/mean ratio) to consider signal present; 0 disables (default 3)
    -st int
    	speed threshold (%) to consider a tone valid (default 75)
    -threshold int
    	Ratio (%) between min and max signal level to be considered a valid tone (default 50)
    -verbose
    	log segment durations and dit/dah classifications to stderr
    -wpm int
    	words per minute (for timing) (default 20)

<img width="1594" height="766" alt="screenshot" src="https://github.com/user-attachments/assets/893de979-9911-44d2-9c9d-2b22626398a6" />
