# mode
Morse Decoder

A morse decoder that can decode audio from a device (i.e. SDR or ham radio) or a file (wave files)

Usage: mode [options] [filename]

    -bandwidth float
    	bandwidth for bandpass filter (in Hz) (default 300)
    -buffer int
    	buffer size (in ms) (default 300)
    -device string
    	input audio device (for live decoding)
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
    -noisegate float
    	Noise gate (squelch) level (0.0-1.0) (default 0.2)
    -noui
    	no user interface, write to stdout
    -play string
    	output audio device (for monitoring)
    -separator
    	output separator '_' between decoded segments
    -threshold int
    	Ratio (%) between min and max signal level to be considered a valid tone (default 50)
    -wpm int
    	words per minute (for timing) (default 20)

<img width="1594" height="766" alt="screenshot" src="https://github.com/user-attachments/assets/893de979-9911-44d2-9c9d-2b22626398a6" />
