ARGS=

all: MoDe.app

MoDe.app: mode Icon.png
	# create Mode.app
	fyne package -os darwin -exe mode -icon Icon.png -name MoDe
	# request permission for using microphone (and other audio inputs)
	./set-permissions.sh MoDe.app/Contents/Info.plist

mode: gmain.go decoder.go
	go build -o mode gmain.go decoder.go

clean:
	rm -rf MoDe.app mode

gui:
	go run gmain.go decoder.go $(ARGS)

tui:
	go run main.go decoder.go $(ARGS)

text:
	go run main.go decoder.go -noui $(ARGS)
