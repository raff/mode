//go:build nobuild

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/gordonklaus/portaudio"
	"github.com/j-04/gocui-component"
	"github.com/jroimartin/gocui"
)

type App struct {
	DecoderApp

	gui   *gocui.Gui
	vinfo *gocui.View
	vmain *gocui.View
	vcmd  *gocui.View

	startTime time.Time
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

		if app.Player != nil {
			fmt.Fprintf(app.vcmd, "  v: -volume")
		}

		if app.Player != nil {
			fmt.Fprintf(app.vcmd, "  f: toggle filter      -: toggle separator\n")
		} else {
			fmt.Fprintf(app.vcmd, "  f: toggle filter\n")
		}

		fmt.Fprintf(app.vcmd, "c: clear     B: +bandwidth  W: +wpm  N: +noise T: +threshold")

		if app.Player != nil {
			fmt.Fprintf(app.vcmd, "  V: +volume  m: toggle audio/mute")
		}
	}

	d := time.Since(app.startTime)
	app.vinfo.Clear()
	app.vinfo.SetOrigin(0, 0)

	fwpm := app.Mode.fwpm
	if fwpm <= 0 {
		fwpm += app.Mode.wpm
	}

	fmt.Fprintf(app.vinfo,
		"[%v] Tone: %3dhz Filter: %-4s  WPM:%2d/%2d (%2d) dit:%-2dms sp:%-2d/%-3dms  NG:%3.1f  Thr:%2d%%  Bw:%3d  Level:%3d (T:%3d S:%3d)",
		string(app.Spectrogram[:]),
		app.Tone,
		app.Fname,
		app.Mode.wpm,
		fwpm,
		1200/app.Mode.ditTime,
		app.Mode.ditTime,
		app.Mode.mSpace,
		app.Mode.wSpace,
		app.NoiseGate,
		app.Threshold,
		int(app.Bandwidth),
		int(app.Mag*1000),
		int(app.Mode.tmag), // *1000),
		int(app.Mode.smag), // *1000),
	)

	if app.Player != nil {
		fmt.Fprintf(app.vinfo, "   %8v  vol: %d",
			d.Truncate(time.Second).String(),
			int(app.Player.Volume*10))
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
	// bandwidth up/down: B / b
	//

	bandwidthUp := func(g *gocui.Gui, v *gocui.View) error {
		if app.Bandwidth < 500 {
			app.Bandwidth += 50
		}

		return nil
	}

	bandwidthDown := func(g *gocui.Gui, v *gocui.View) error {
		if app.Bandwidth > 50 {
			app.Bandwidth -= 50
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
		if app.Mode.wpm < 50 {
			app.Mode.wpm++
		}

		return nil
	}

	wpmDown := func(g *gocui.Gui, v *gocui.View) error {
		if app.Mode.wpm > 1 {
			app.Mode.wpm--
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
	// Farnsworth timing up/down: T / t
	//

	fwpmUp := func(g *gocui.Gui, v *gocui.View) error {
		if v := app.Mode.getFwpm(); v < 50 {
			app.Mode.setFwpm(v + 1)
		}

		return nil
	}

	fwpmDown := func(g *gocui.Gui, v *gocui.View) error {
		if v := app.Mode.getFwpm(); v > 1 {
			app.Mode.setFwpm(v - 1)
		}

		return nil
	}

	if err := app.gui.SetKeybinding("", 'S', gocui.ModNone, fwpmUp); err != nil {
		return err
	}

	if err := app.gui.SetKeybinding("", 's', gocui.ModNone, fwpmDown); err != nil {
		return err
	}

	//
	// noise gate up/down: N / n
	//

	noiseUp := func(g *gocui.Gui, v *gocui.View) error {
		if app.NoiseGate < 1 {
			app.NoiseGate += 0.1
		}

		return nil
	}

	noiseDown := func(g *gocui.Gui, v *gocui.View) error {
		if app.NoiseGate > 0.1 {
			app.NoiseGate -= 0.1
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
		if app.Threshold < 100 {
			app.Threshold += 5
		}

		return nil
	}

	thresholdDown := func(g *gocui.Gui, v *gocui.View) error {
		if app.Threshold > 5 {
			app.Threshold -= 5
		}

		return nil
	}

	if err := app.gui.SetKeybinding("", 'T', gocui.ModNone, thresholdUp); err != nil {
		return err
	}

	if err := app.gui.SetKeybinding("", 't', gocui.ModNone, thresholdDown); err != nil {
		return err
	}

	//
	// toggle filter: f
	//

	toggleFilter := func(g *gocui.Gui, v *gocui.View) error {
		switch app.Fname {
		case "no":
			app.Fname = "bp"
			app.Filter = Denoise
		case "bp":
			app.Fname = "apf"
			app.Filter = AudioPeakFilter(1.413)
		case "apf":
			app.Fname = "apf2"
			app.Filter = AudioPeakFilter(2)
		case "apf2":
			app.Fname = "no"
			app.Filter = nil
		default:
			// Default to bandpass if unknown
			app.Fname = "bp"
			app.Filter = Denoise
		}
		return nil
	}

	if err := app.gui.SetKeybinding("", 'f', gocui.ModNone, toggleFilter); err != nil {
		return err
	}

	if app.Player != nil {

		//
		// toggle mute: m
		//

		toggleMute := func(g *gocui.Gui, v *gocui.View) error {
			app.Mute = !app.Mute
			app.Player.Mute(app.Mute)
			return nil
		}

		if err := app.gui.SetKeybinding("", 'm', gocui.ModNone, toggleMute); err != nil {
			return err
		}

		//
		// volume up/down: V / v
		//

		volumeUp := func(g *gocui.Gui, v *gocui.View) error {
			if app.Player.Volume < 2.0 {
				app.Player.Volume += 0.1
			}

			return nil
		}

		volumeDown := func(g *gocui.Gui, v *gocui.View) error {
			if app.Player.Volume > 0.0 {
				app.Player.Volume -= 0.1
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

func (app *App) addText(s string) {
	if app.gui == nil {
		fmt.Print(s)
		return
	}

	app.gui.Update(func(g *gocui.Gui) error {
		fmt.Fprint(app.vmain, s)
		return nil
	})
}

var (
	FormSelect = fmt.Errorf("form-selected")
	FormCancel = fmt.Errorf("form-cancel")
)

func guiSelectAudio(ssize int) (reader *AudioReader) {
	g, err := gocui.NewGui(gocui.OutputNormal)
	if err != nil {
		log.Panicln(err)
	}
	defer g.Close()

	list, err := ListAudioDevices(AudioIn)
	if err != nil {
		log.Fatal(err)
	}

	form := component.NewForm(g, "Select input device", 8, len(list), 0, 0)
	sel := form.AddSelect("Device:", 8, 40).AddOptions(list...)

	form.AddButton("Select", func(g *gocui.Gui, v *gocui.View) error {
		reader, err = FromAudioStream(sel.GetSelected(), ssize)
		if err != nil {
			log.Fatal(err)
		}

		form.Close(g, v)
		return FormSelect
	})

	form.AddButton("Cancel", func(g *gocui.Gui, v *gocui.View) error {
		form.Close(g, v)
		return FormCancel
	})

	form.Draw()

	if err := g.MainLoop(); err != FormSelect && err != FormCancel {
		log.Panicln(err)
	}

	return
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
	} else if *noui {
		log.Fatal("no input source specified")
	} else {
		reader = guiSelectAudio(*ssize)
	}

	if *out != "" {
		player, err = NewAudioWriter(*out, reader.SampleRate, *ssize)
		if err != nil {
			log.Fatal(err)
		}
	}

	var g *gocui.Gui

	if *noui == false {
		g, err = gocui.NewGui(gocui.OutputNormal)
		if err != nil {
			log.Panicln(err)
		}
		defer g.Close()
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

	app := App{
		gui:       g,
		startTime: time.Now(),

		DecoderApp: DecoderApp{
			Bandwidth: *bandwidth,
			Threshold: *threshold,
			NoiseGate: *noiseGate,
			MinFreq:   *minFreq,
			MaxFreq:   *maxFreq,
			Reader:    reader,
			Player:    player,
			Mode:      NewMorseDecoder(*wpm, *fwpm, float64(*st)/100),
			Filter:    af,
			Fname:     *filter,
		},
	}

	app.DecoderApp.AddText = app.addText

	if app.Reader == nil {
		log.Fatal("No audio selected")
	}

	if g != nil {
		g.SetManagerFunc(app.Layout)
		app.SetKeyBinding()

		go app.MainLoop()

		if err := g.MainLoop(); err != nil && err != gocui.ErrQuit {
			log.Panicln(err)
		}

		return
	}

	app.MainLoop()
}
