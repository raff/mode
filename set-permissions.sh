#!/bin/sh
exec sed -i '' '/<\/dict>/i \
	<key>NSMicrophoneUsageDescription</key> \
	<string>Mode requires access to the microphone in order to decode.</string>' $1
