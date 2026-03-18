// Package session writes decoded Morse text to a dated log file.
package session

import (
	"bufio"
	"log"
	"os"
	"path/filepath"
	"time"
)

// Log writes decoded text to ~/mode-sessions/YYYY-MM-DD_HH-MM.txt.
// Call Close() when the session ends to flush the buffer.
type Log struct {
	f   *os.File
	bw  *bufio.Writer
}

// Open creates (or appends to) the session log file for the current time.
// It returns a no-op Log and logs a warning if the file cannot be opened.
func Open() *Log {
	home, err := os.UserHomeDir()
	if err != nil {
		log.Printf("session log: home dir: %v", err)
		return &Log{}
	}

	dir := filepath.Join(home, "mode-sessions")
	if err := os.MkdirAll(dir, 0700); err != nil {
		log.Printf("session log: mkdir: %v", err)
		return &Log{}
	}

	name := time.Now().Format("2006-01-02_15-04") + ".txt"
	path := filepath.Join(dir, name)

	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		log.Printf("session log: open %s: %v", path, err)
		return &Log{}
	}

	return &Log{f: f, bw: bufio.NewWriter(f)}
}

// Write appends text to the session log. Safe to call with a nil/no-op Log.
func (l *Log) Write(s string) {
	if l.bw == nil {
		return
	}
	_, err := l.bw.WriteString(s)
	if err != nil {
		log.Printf("session log: write: %v", err)
	}
}

// Close flushes the buffer and closes the underlying file.
func (l *Log) Close() {
	if l.bw != nil {
		if err := l.bw.Flush(); err != nil {
			log.Printf("session log: flush: %v", err)
		}
	}
	if l.f != nil {
		l.f.Close()
	}
}
