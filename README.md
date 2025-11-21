# mode
Morse Decoder

A morse decoder that can decode audio from a device (i.e. SDR or ham radio) or a file (wave files)

Usage: mode [options] [filename]

   -buffer int
    	buffer size (in ms) (default 300)

   -device string
    	input audio device (for live decoding)

   -threshold int
    	Threshold ration (percentage) (default 50)

   -wpm int
    	words per minute (for timing) (default 20)

