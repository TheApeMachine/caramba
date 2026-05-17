package weights

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	pathpkg "path"
	"path/filepath"
	"strings"
)

const defaultConfigFile = "config.json"

/*
ResolveConfig reads the Hugging Face-style config.json sidecar for a
SafeTensors source. Local directories, local files, and Hub sources use
the same Source fields as Resolve.
*/
func ResolveConfig(ctx context.Context, source Source) (map[string]any, error) {
	source.Source = strings.TrimSpace(source.Source)

	if source.Source == "" {
		return nil, fmt.Errorf("weights: source is required")
	}

	if path, err := source.localConfigPath(); err == nil {
		return readConfig(path)
	}

	filename := source.configFilename()
	path, err := source.download(ctx, filename)

	if err != nil {
		return nil, fmt.Errorf("weights: download %s: %w", filename, err)
	}

	return readConfig(path)
}

func (source Source) localConfigPath() (string, error) {
	info, err := os.Stat(source.Source)

	if err != nil {
		return "", err
	}

	if strings.TrimSpace(source.File) != "" {
		if !info.IsDir() {
			return "", fmt.Errorf(
				"weights: source %s is a file but explicit file %s was requested",
				source.Source,
				source.File,
			)
		}

		return filepath.Join(source.Source, filepath.FromSlash(source.configFilename())), nil
	}

	if info.IsDir() {
		return filepath.Join(source.Source, defaultConfigFile), nil
	}

	return filepath.Join(filepath.Dir(source.Source), defaultConfigFile), nil
}

func (source Source) configFilename() string {
	filename := filepath.ToSlash(strings.TrimSpace(source.File))

	if filename == "" {
		return defaultConfigFile
	}

	base := pathpkg.Dir(filename)

	if base == "." || base == "/" {
		return defaultConfigFile
	}

	return pathpkg.Join(base, defaultConfigFile)
}

func readConfig(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)

	if err != nil {
		return nil, fmt.Errorf("weights: read config %s: %w", path, err)
	}

	var config map[string]any

	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("weights: parse config %s: %w", path, err)
	}

	return config, nil
}
