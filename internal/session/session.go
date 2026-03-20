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
	f  *os.File
	bw *bufio.Writer
}

// Dir returns the path to the sessions directory (~/.mode-sessions).
func Dir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, "mode-sessions")
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	return dir, nil
}

// open is the shared implementation for Open and OpenAt.
func open(t time.Time, format string) *Log {
	dir, err := Dir()
	if err != nil {
		log.Printf("session log: dir: %v", err)
		return &Log{}
	}

	name := t.Format(format) + ".txt"
	path := filepath.Join(dir, name)

	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		log.Printf("session log: open %s: %v", path, err)
		return &Log{}
	}

	return &Log{f: f, bw: bufio.NewWriter(f)}
}

// Open creates (or appends to) the session log file for the current time
// (minute-level granularity). It returns a no-op Log and logs a warning
// if the file cannot be opened.
func Open() *Log {
	return open(time.Now(), "2006-01-02_15-04")
}

// OpenAt creates a session log file for the given time using second-level
// granularity, so it always produces a unique file for a recording session.
func OpenAt(t time.Time) *Log {
	return open(t, "2006-01-02_15-04-05")
}

// Path returns the file path of the session log, or "" if not open.
func (l *Log) Path() string {
	if l.f == nil {
		return ""
	}
	return l.f.Name()
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
