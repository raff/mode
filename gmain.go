package main

import (
	_ "embed"
	"flag"
	"fmt"
	"image/color"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gordonklaus/portaudio"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/storage"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
	"fyne.io/x/fyne/wrapper"
)

//go:embed assets/fonts/UbuntuMono-R.ttf
var fontBits_R []byte
var font_R = &fyne.StaticResource{StaticName: "UbuntuMono-R.ttf", StaticContent: fontBits_R}

//go:embed assets/fonts/UbuntuMono-RI.ttf
var fontBits_RI []byte
var font_RI = &fyne.StaticResource{StaticName: "UbuntuMono-RI.ttf", StaticContent: fontBits_RI}

//go:embed assets/fonts/UbuntuMono-B.ttf
var fontBits_B []byte
var font_B = &fyne.StaticResource{StaticName: "UbuntuMono-B.ttf", StaticContent: fontBits_B}

//go:embed assets/fonts/UbuntuMono-BI.ttf
var fontBits_BI []byte
var font_BI = &fyne.StaticResource{StaticName: "UbuntuMono-BI.ttf", StaticContent: fontBits_BI}

type CompactTheme struct{}

var _ fyne.Theme = (*CompactTheme)(nil)

func (t CompactTheme) Font(style fyne.TextStyle) fyne.Resource {
	if style.Bold && style.Italic {
		return font_BI
	}

	if style.Bold {
		return font_B
	}

	if style.Italic {
		return font_RI
	}

	return font_R
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

	if name == theme.SizeNameText {
		return theme.DefaultTheme().Size(name) * 1.5
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
			if num < s.vmin {
				num = s.vmin
			} else if num > s.vmax {
				num = s.vmax
			}

			if num != s.value {
				s.value = num
				if s.onChange != nil {
					s.onChange(s.value)
				}
			}
		}

		// update to new value or reset to original value
		s.updateDisplay()
	}

	upIcon := widget.NewIcon(theme.Icon(theme.IconNameArrowDropUp))
	upButton := wrapper.MakeTappable(upIcon, func(e *fyne.PointEvent) {
		s.Increment()
	})

	downIcon := widget.NewIcon(theme.Icon(theme.IconNameArrowDropDown))
	downButton := wrapper.MakeTappable(downIcon, func(e *fyne.PointEvent) {
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

// TextLog is a wrapper around widget.RichText that supports appending text
type TextLog struct {
	*widget.RichText
}

// NewTextLog creates a new TextLog with initial text
func NewTextLog(text string) *TextLog {
	t := &TextLog{
		RichText: widget.NewRichTextWithText(text),
	}
	t.ExtendBaseWidget(t)
	return t
}

// Append adds text to the log.
// If text contains a space or the last segment isn't text, it adds a new TextSegment.
func (t *TextLog) Append(text string) {
	parts := strings.Split(text, " ")
	if len(parts) == 0 {
		return
	}

	if len(t.Segments) > 0 {
		if seg, ok := t.Segments[len(t.Segments)-1].(*widget.TextSegment); ok {
			seg.Text += parts[0]
			parts = parts[1:]
		}
	}

	for _, p := range parts {
		t.Segments = append(t.Segments, &widget.TextSegment{
			Text:  p + " ",
			Style: widget.RichTextStyleInline,
		})
	}

	t.Refresh()
}

func (t *TextLog) Clear() {
	t.Segments = t.Segments[:0]
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
		// log.Fatal("no input source specified")
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
		*filter = "Bandpass"
		af = Denoise
	case "apf":
		*filter = "APF"
		af = AudioPeakFilter(1.413)
	case "apf2":
		*filter = "APF 2"
		af = AudioPeakFilter(2)
	default:
		*filter = "None"
	}

	var modeApp DecoderApp

	// Create a new application
	myApp := app.New()
	myApp.Settings().SetTheme(&CompactTheme{})
	myWindow := myApp.NewWindow("Morse Decoder")

	boldText := func(s string) *widget.Label {
		return widget.NewLabelWithStyle(s, fyne.TextAlignLeading, fyne.TextStyle{Bold: true, Monospace: true})
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

	var fw int
	if *fwpm == 0 || *fwpm >= *wpm {
		fw = *wpm
	} else if *fwpm < 0 {
		fw = *wpm + *fwpm
	} else {
		fw = *fwpm
	}

	fwpmStepper := NewNumericStepper(5, 50, fw, 1, func(value int) {
		modeApp.Mode.setFwpm(value)
	})

	// Create a numeric stepper widget (min: 0, max: 100, initial: 50)
	wpmStepper := NewNumericStepper(5, 50, *wpm, 1, func(value int) {
		modeApp.Mode.wpm = value

		fwpmStepper.SetValue(modeApp.Mode.getFwpm())
	})

	filterSel := widget.NewSelect([]string{"None", "Bandpass", "APF", "APF 2"}, func(selected string) {
		switch selected {
		case "None":
			modeApp.Filter = nil

		case "Bandpass":
			modeApp.Filter = Denoise

		case "APF":
			modeApp.Filter = AudioPeakFilter(1.413)

		case "APF 2":
			modeApp.Filter = AudioPeakFilter(2)
		}
	})

	filterSel.SetSelected(*filter)

	freqLabel := boldText("100")
	audiospectrum := boldText("")
	calcWpm := boldText("(00)")
	textOut := NewTextLog("")
	textOut.Wrapping = fyne.TextWrapWord
	textOut.Scroll = container.ScrollVerticalOnly

	deviceBtn := widget.NewButtonWithIcon("", theme.MediaMusicIcon(), func() {
		l, err := ListAudioDevices(AudioIn)
		if err != nil {
			dialog.ShowError(err, myWindow)
			return
		}

		deviceSel := widget.NewSelect(l, func(selected string) {
			log.Println("device selected:", selected)
		})

		if modeApp.Reader != nil {
			deviceSel.SetSelected(modeApp.Reader.Id)
		}

		deviceDialog := dialog.NewForm(
			"Select audio device",
			"Select",
			"Cancel",
			[]*widget.FormItem{
				{Text: "Devices", Widget: deviceSel},
			},
			func(submitted bool) {
				log.Println("select audio device", submitted)

				if !submitted {
					return
				}

				modeApp.SetReader(nil)

				ar, err := FromAudioStream(deviceSel.Selected, *ssize)
				if err != nil {
					dialog.ShowError(err, myWindow)
					return
				}

				// do something with ar
				modeApp.SetReader(ar)
			},
			myWindow,
		)

		deviceDialog.Show()
	})

	fileBtn := widget.NewButtonWithIcon("", theme.FileAudioIcon(), func() {
		fd := dialog.NewFileOpen(func(reader fyne.URIReadCloser, err error) {
			if err != nil {
				dialog.ShowError(err, myWindow)
				return
			}
			if reader == nil {
				return
			}

			modeApp.SetReader(nil)

			wr, err := FromWaveFile(reader.(io.ReadSeeker), 1)
			if err != nil {
				dialog.ShowError(err, myWindow)
				return
			}

			log.Println("file selected:", reader.URI())

			// do something with wr
			modeApp.SetReader(wr)
		}, myWindow)
		fd.SetFilter(storage.NewExtensionFileFilter([]string{".wav"}))
		fd.Show()
	})

	// Create the toolbar container
	toolbar := container.NewHBox(
		deviceBtn,
		fileBtn,
		withBorder(audiospectrum),
		boldText("Freq:"), freqLabel,
		boldText("Filter:"), filterSel,
		boldText("WPM:"), wpmStepper,
		boldText("FWPM:"), fwpmStepper,
		calcWpm,
	)

	clearBtn := widget.NewButton("Clear", func() {
		textOut.Clear()
	})

	quitBtn := widget.NewButtonWithIcon("", theme.LogoutIcon(), func() {
		myApp.Quit()
	})

	bottom := container.NewHBox(
		clearBtn,
		statusLabel,
		layout.NewSpacer(),
		quitBtn,
	)

	// Arrange the widgets in a vertical container
	content := container.NewBorder(
		withBorder(toolbar), // top
		withBorder(bottom),  // bottom
		nil,                 // left
		nil,                 // right
		withBorder(textOut), // middle
	)

	modeApp = DecoderApp{
		Wait: true,

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
				textOut.Append(s)
			})
		},
	}

	modeApp.Update = func() {
		fyne.Do(func() {
			freqLabel.SetText(fmt.Sprintf("%-3d", modeApp.Tone))
			calcWpm.SetText(fmt.Sprintf("(%2d) dit:%-2dms sp:%-2d/%-3dms",
				1200/modeApp.Mode.ditTime,
				modeApp.Mode.ditTime,
				modeApp.Mode.mSpace,
				modeApp.Mode.wSpace))
			audiospectrum.SetText(string(modeApp.Spectrogram[:]))
		})
	}

	go modeApp.MainLoop()

	// Set the content and show the window
	myWindow.SetContent(content)
	myWindow.Resize(fyne.NewSize(600, 400))
	myWindow.ShowAndRun()
}
