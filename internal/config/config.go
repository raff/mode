package config

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// Config holds persistent settings for the MoDe application.
// Session-specific values (wpm, fwpm, wav path) are intentionally excluded.
type Config struct {
	Device    string  `json:"device,omitempty"`    // last selected audio input device
	Filter    string  `json:"filter,omitempty"`    // filter name: "None", "Bandpass", "APF", "APF 2"
	Squelch   int     `json:"squelch,omitempty"`   // spectral peak ratio (0–5)
	Bandwidth float64 `json:"bandwidth,omitempty"` // bandpass filter bandwidth in Hz
	MinSNR    float64 `json:"min_snr,omitempty"`   // minimum SNR (envelope peak − floor) after normalization to gate noise
}

func configPath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "mode", "config.json"), nil
}

// Load reads the saved config. Returns a zero-value Config (not an error) if the
// file does not exist yet.
func Load() (Config, error) {
	path, err := configPath()
	if err != nil {
		return Config{}, err
	}

	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return Config{}, nil
	}
	if err != nil {
		return Config{}, err
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, err
	}

	return cfg, nil
}

// Save writes the config file, creating the parent directory if needed.
func Save(cfg Config) error {
	path, err := configPath()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0600)
}
