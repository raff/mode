//go:build nobuild

package main

import (
	"flag"
	"fmt"
	"image/color"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"github.com/gordonklaus/portaudio"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
	"fyne.io/x/fyne/wrapper"
)

type CompactTheme struct{}

var _ fyne.Theme = (*CompactTheme)(nil)

func (t CompactTheme) Font(style fyne.TextStyle) fyne.Resource {
	return theme.DefaultTheme().Font(style)
}

func (t CompactTheme) Color(name fyne.ThemeColorName, variant fyne.ThemeVariant) color.Color {
	return theme.DefaultTheme().Color(name, variant)
}

func (t CompactTheme) Icon(name fyne.ThemeIconName) fyne.Resource {
	return theme.DefaultTheme().Icon(name)
}

func (t CompactTheme) Size(name fyne.ThemeSizeName) float32 {
	if name == theme.SizeNamePadding {
		return 2
	}

	return theme.DefaultTheme().Size(name)
}

// BorderedContainer is a container that draws a rounded border around its content
type BorderedContainer struct {
	widget.BaseWidget
	content      fyne.CanvasObject
	borderColor  color.Color
	borderRadius float32
	borderWidth  float32
	padding      float32
}

// NewBorderedContainer creates a new container with a rounded border
func NewBorderedContainer(content fyne.CanvasObject, borderColor color.Color, borderRadius, borderWidth, padding float32) *BorderedContainer {
	bc := &BorderedContainer{
		content:      content,
		borderColor:  borderColor,
		borderRadius: borderRadius,
		borderWidth:  borderWidth,
		padding:      padding,
	}
	bc.ExtendBaseWidget(bc)
	return bc
}

// CreateRenderer returns the renderer for the bordered container
func (bc *BorderedContainer) CreateRenderer() fyne.WidgetRenderer {
	// Create the rounded rectangle for the border
	border := canvas.NewRectangle(bc.borderColor)
	border.StrokeColor = bc.borderColor
	border.StrokeWidth = bc.borderWidth
	border.FillColor = color.Transparent
	border.CornerRadius = bc.borderRadius

	// Wrap content with padding
	paddedContent := container.NewPadded(bc.content)

	return &borderedContainerRenderer{
		border:  border,
		content: paddedContent,
		objects: []fyne.CanvasObject{border, paddedContent},
		bc:      bc,
	}
}

type borderedContainerRenderer struct {
	border  *canvas.Rectangle
	content *fyne.Container
	objects []fyne.CanvasObject
	bc      *BorderedContainer
}

func (r *borderedContainerRenderer) Layout(size fyne.Size) {
	// Border takes the full size
	r.border.Resize(size)
	r.border.Move(fyne.NewPos(0, 0))

	// Content is inset by padding and border
	inset := r.bc.padding + r.bc.borderWidth
	contentSize := fyne.NewSize(
		size.Width-inset*2,
		size.Height-inset*2,
	)
	r.content.Resize(contentSize)
	r.content.Move(fyne.NewPos(inset, inset))
}

func (r *borderedContainerRenderer) MinSize() fyne.Size {
	contentMin := r.content.MinSize()
	inset := (r.bc.padding + r.bc.borderWidth) * 2
	return fyne.NewSize(
		contentMin.Width+inset,
		contentMin.Height+inset,
	)
}

func (r *borderedContainerRenderer) Refresh() {
	r.border.StrokeColor = r.bc.borderColor
	r.border.StrokeWidth = r.bc.borderWidth
	r.border.CornerRadius = r.bc.borderRadius
	r.border.Refresh()
	canvas.Refresh(r.bc)
}

func (r *borderedContainerRenderer) Objects() []fyne.CanvasObject {
	return r.objects
}

func (r *borderedContainerRenderer) Destroy() {}

// NumericStepper is a custom widget that allows incrementing/decrementing a number
type NumericStepper struct {
	widget.BaseWidget
	value     int
	vmin      int
	vmax      int
	step      int
	entry     *widget.Entry
	onChange  func(int)
	container *fyne.Container
}

// NewNumericStepper creates a new numeric stepper widget
func NewNumericStepper(min, max, initial, step int, onChange func(int)) *NumericStepper {
	stepper := &NumericStepper{
		value:    initial,
		vmin:     min,
		vmax:     max,
		step:     step,
		onChange: onChange,
	}
	stepper.ExtendBaseWidget(stepper)
	stepper.buildUI()
	return stepper
}

// buildUI constructs the stepper interface
func (s *NumericStepper) buildUI() {
	// Create entry field for manual input with compact size
	s.entry = widget.NewEntry()
	s.entry.SetText(fmt.Sprintf("%d", s.value))

	// Calculate width based on max value digits (add some padding)
	maxDigits := len(fmt.Sprintf("%d", s.vmax))
	entryWidth := float32(maxDigits*14 + 20) // approximate width per digit + padding
	s.entry.Resize(fyne.NewSize(entryWidth, 30))

	s.entry.OnChanged = func(content string) {
		if content == "" {
			return
		}
		if num, err := strconv.Atoi(content); err == nil {
			if num >= s.vmin && num <= s.vmax {
				s.value = num
				if s.onChange != nil {
					s.onChange(s.value)
				}
			} else if num < s.vmin {
				s.entry.SetText(fmt.Sprintf("%d", s.vmin))
			} else if num > s.vmax {
				s.entry.SetText(fmt.Sprintf("%d", s.vmax))
			}
		} else {
			// Reset to current value if invalid
			s.entry.SetText(fmt.Sprintf("%d", s.value))
		}
	}

	upIcon := widget.NewIcon(theme.Icon(theme.IconNameArrowDropUp))
	upButton := wrapper.MakeTappable(upIcon, func(e *fyne.PointEvent) {
		fmt.Println("Up button tapped")
		s.Increment()
	})

	downIcon := widget.NewIcon(theme.Icon(theme.IconNameArrowDropDown))
	downButton := wrapper.MakeTappable(downIcon, func(e *fyne.PointEvent) {
		fmt.Println("Down button tapped")
		s.Decrement()
	})

	// Create compact container with fixed widths
	entryContainer := container.NewStack(s.entry)
	buttonContainer := container.New(layout.NewCustomPaddedVBoxLayout(0), upButton, downButton)

	// Arrange entry and buttons horizontally with compact sizing
	s.container = container.NewHBox(
		entryContainer,
		buttonContainer,
	)
}

// Increment increases the value by step
func (s *NumericStepper) Increment() {
	if s.value < s.vmax {
		s.value += s.step
		s.updateDisplay()
		if s.onChange != nil {
			s.onChange(s.value)
		}
	}
}

// Decrement decreases the value by step
func (s *NumericStepper) Decrement() {
	if s.value > s.vmin {
		s.value -= s.step
		s.updateDisplay()
		if s.onChange != nil {
			s.onChange(s.value)
		}
	}
}

// updateDisplay updates the displayed value
func (s *NumericStepper) updateDisplay() {
	s.entry.SetText(fmt.Sprintf("%d", s.value))
}

// GetValue returns the current value
func (s *NumericStepper) GetValue() int {
	return s.value
}

// SetValue sets the value programmatically
func (s *NumericStepper) SetValue(value int) {
	if value >= s.vmin && value <= s.vmax {
		s.value = value
		s.updateDisplay()
		if s.onChange != nil {
			s.onChange(s.value)
		}
	}
}

// CreateRenderer returns the widget renderer
func (s *NumericStepper) CreateRenderer() fyne.WidgetRenderer {
	return widget.NewSimpleRenderer(s.container)
}

func main() {
	ssize := flag.Int("buffer", 300, "buffer size (in ms)")
	wpm := flag.Int("wpm", 20, "words per minute (for timing)")
	fwpm := flag.Int("fwpm", 0, "Farnsworth speed")
	dev := flag.String("device", "", "input audio device (for live decoding)")
	out := flag.String("play", "", "output audio device (for monitoring)")
	list := flag.Bool("list", false, "list audio devices")
	bandwidth := flag.Float64("bandwidth", 300, "bandwidth for bandpass filter (in Hz)")
	noiseGate := flag.Float64("noisegate", 0.2, "Noise gate (squelch) level (0.0-1.0)")
	threshold := flag.Int("threshold", 50, "Ratio (%) between min and max signal level to be considered a valid tone")
	st := flag.Int("st", 75, "speed threshold (%) to consider a tone valid")
	filter := flag.String("filter", "bp", "apply bandpass filter (bp), audio peak filter (apf), or no filter (none)")
	minFreq := flag.Float64("minfreq", 300.0, "minimum frequency (in Hz)")
	maxFreq := flag.Float64("maxfreq", 2000.0, "maximum frequency (in Hz)")
	noui := flag.Bool("noui", false, "no user interface, write to stdout")

	flag.Parse()

	if *threshold < 1 {
		*threshold = 1
	}
	if *threshold > 100 {
		*threshold = 100
	}
	if *st < 1 {
		*st = 1
	}
	if *st > 100 {
		*st = 100
	}

	if *dev != "" || *out != "" || flag.NArg() == 0 {
		// Initialize PortAudio
		err := portaudio.Initialize()
		if err != nil {
			log.Fatalf("Failed to initialize PortAudio: %v", err)
		}
		defer portaudio.Terminate()
	}

	if *list || (*noui && *dev == "" && flag.NArg() == 0) {
		fmt.Println()
		fmt.Printf("Usage: %v [options] [filename]\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
		fmt.Println()

		l, err := ListAudioDevices(AudioInOut)
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

	var af AudioFilter
	switch *filter {
	case "bp":
		af = Denoise
	case "apf":
		af = AudioPeakFilter(1.413)
	case "apf2":
		af = AudioPeakFilter(2)
	default:
		*filter = "no"
	}

	// Create a new application
	myApp := app.New()
	myApp.Settings().SetTheme(&CompactTheme{})
	myWindow := myApp.NewWindow("Morse Decoder")

	boldText := func(s string) *widget.Label {
		return widget.NewLabelWithStyle(s, fyne.TextAlignLeading, fyne.TextStyle{Bold: true})
	}

	withBorder := func(o fyne.CanvasObject) fyne.CanvasObject {
		return NewBorderedContainer(
			o,
			theme.DefaultTheme().Color(theme.ColorNameForeground, myApp.Settings().ThemeVariant()), // border color
			2, // corner radius
			1, // border width
			0, // padding
		)
	}

	statusLabel := boldText("")

	// Create a numeric stepper widget (min: 0, max: 100, initial: 50)
	stepper1 := NewNumericStepper(0, 100, 50, 5, func(value int) {
		statusLabel.SetText(fmt.Sprintf("Current value: %d", value))
	})

	stepper2 := NewNumericStepper(-10, 10, 0, 1, func(value int) {
		statusLabel.SetText(fmt.Sprintf("Current value: %d", value))
	})

	filterSel := widget.NewSelect([]string{"None", "Bandpass", "APF", "APF 2"}, func(selected string) {
		statusLabel.SetText("Selected filter: " + selected)
	})

	filterSel.SetSelectedIndex(0)

	freqLabel := boldText("100")
	audiospectrum := boldText("")
	textGrid := widget.NewTextGrid()

	// Create the toolbar container
	toolbar := container.NewHBox(
		withBorder(audiospectrum),
		boldText("Freq:"), freqLabel,
		boldText("Filter:"), filterSel,
		boldText("WPM:"), stepper1,
		boldText("FWPM:"), stepper2)

	// Arrange the widgets in a vertical container
	content := container.NewBorder(
		withBorder(toolbar),     // top
		withBorder(statusLabel), // bottom
		nil,                     // left
		nil,                     // right
		withBorder(textGrid),    // middle
	)

	modeApp := DecoderApp{
		Bandwidth: *bandwidth,
		Threshold: *threshold,
		NoiseGate: *noiseGate,
		MinFreq:   *minFreq,
		MaxFreq:   *maxFreq,
		Reader:    reader,
		Player:    player,
		Mode:      NewMorseDecoder(*wpm, *fwpm, float64(*st)/100),
		Filter:    af,
		SetStatus: func(s string) {
			fyne.Do(func() {
				statusLabel.SetText(s)
			})
		},
		AddText: func(s string) {
			fyne.Do(func() {
				textGrid.Append(s)
			})
		},
	}

	modeApp.Update = func() {
		fyne.Do(func() {
			freqLabel.SetText(strconv.Itoa(modeApp.Tone))
			audiospectrum.SetText(string(modeApp.Spectrogram[:]))
		})
	}

	go modeApp.MainLoop()

	// Set the content and show the window
	myWindow.SetContent(content)
	myWindow.Resize(fyne.NewSize(600, 400))
	myWindow.ShowAndRun()
}
