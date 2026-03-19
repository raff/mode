ARGS=

help:
	@echo
	@echo Available targets
	@echo
	@echo "  gmode - build GUI app"
	@echo "  tmode - build TUI app"
	@echo "  gui   - run GUI app"
	@echo "  tui   - run TUI app"
	@echo "  text  - run TUI app, not interactive"
	@echo "  app   - build MacOS app (MoDe.app)"
	@echo "  clean - cleanup"
	@echo

app: MoDe.app

MoDe.app: gmode tmode cmd/gmode/Icon.png
	# create Mode.app
	fyne package -os darwin -sourceDir cmd/gmode -exe gmode -icon Icon.png -name MoDe
	# copy TUI executable, just in case
	cp tmode MoDe.app/Contents/MacOS/tmode
	# request permission for using microphone (and other audio inputs)
	./set-permissions.sh MoDe.app/Contents/Info.plist

gmode: cmd/gmode/main.go internal/decoder/decoder.go
	go build -o gmode ./cmd/gmode

tmode: cmd/tmode/main.go internal/decoder/decoder.go
	go build -o tmode ./cmd/tmode

clean:
	rm -rf MoDe.app gmode tmode

gui:
	go run ./cmd/gmode $(ARGS)

tui:
	go run ./cmd/tmode $(ARGS)

text:
	go run ./cmd/tmode -noui $(ARGS)
