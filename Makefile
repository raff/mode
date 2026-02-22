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

MoDe.app: gmode tmode Icon.png
	# create Mode.app
	fyne package -os darwin -exe gmode -icon Icon.png -name MoDe
	# copy TUI executable, just in case
	cp tmode MoDe.app/Contents/MacOS/tmode
	# request permission for using microphone (and other audio inputs)
	./set-permissions.sh MoDe.app/Contents/Info.plist

gmode: gmain.go decoder.go
	go build -o gmode gmain.go decoder.go

tmode: main.go decoder.go
	go build -o tmode main.go decoder.go

clean:
	rm -rf MoDe.app gmode tmode

gui:
	go run gmain.go decoder.go $(ARGS)

tui:
	go run main.go decoder.go $(ARGS)

text:
	go run main.go decoder.go -noui $(ARGS)
