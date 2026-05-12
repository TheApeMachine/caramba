package xla

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/theapemachine/caramba/pkg/config"
)

const pjrtHeaderPath = "xla/pjrt/c/pjrt_c_api.h"

/*
PJRTConfig describes the XLA/PJRT files required by the cgo backend.
*/
type PJRTConfig struct {
	Platform    string
	IncludeDir  string
	PluginFile  string
	LibraryDirs []string
}

/*
 NewPJRTConfig reads PJRT paths from the compute section of the project config.

Platform must be one of: cpu, gpu, cuda (cuda aliases to gpu), or tpu.
*/
func NewPJRTConfig(platform string) (PJRTConfig, error) {
	normalizedPlatform, err := normalizedPJRTPlatform(platform)

	if err != nil {
		return PJRTConfig{}, err
	}

	computeConfig := config.NewComputeConfig()

	return PJRTConfig{
		Platform:    normalizedPlatform,
		IncludeDir:  strings.TrimSpace(computeConfig.XLA.IncludeDir),
		PluginFile:  strings.TrimSpace(computeConfig.XLA.PluginFile(normalizedPlatform)),
		LibraryDirs: computeConfig.XLA.LibraryDirs,
	}, nil
}

/*
HeaderFile returns the expected PJRT C API header path.
*/
func (config PJRTConfig) HeaderFile() string {
	if config.IncludeDir == "" {
		return ""
	}

	return filepath.Join(config.IncludeDir, pjrtHeaderPath)
}

/*
PluginName returns the default plugin filename for the configured platform.
*/
func (config PJRTConfig) PluginName() string {
	return config.PluginNames()[0]
}

/*
PluginNames returns accepted plugin filenames for the configured platform.
*/
func (config PJRTConfig) PluginNames() []string {
	if config.Platform == "gpu" {
		return []string{"pjrt_c_api_gpu_plugin.so", "pjrt_c_api_gpu_plugin.dylib"}
	}

	if config.Platform == "tpu" {
		return []string{"pjrt_c_api_tpu_plugin.so", "pjrt_c_api_tpu_plugin.dylib"}
	}

	return []string{"pjrt_c_api_cpu_plugin.so", "pjrt_c_api_cpu_plugin.dylib"}
}

/*
ResolvedPluginFile returns the explicit or discovered PJRT plugin path.
*/
func (config PJRTConfig) ResolvedPluginFile() string {
	if config.PluginFile != "" && pjrtFileExists(config.PluginFile) {
		return config.PluginFile
	}

	for _, libraryDir := range config.LibraryDirs {
		for _, pluginName := range config.PluginNames() {
			candidate := filepath.Join(libraryDir, pluginName)

			if pjrtFileExists(candidate) {
				return candidate
			}
		}
	}

	return ""
}

/*
ValidateBuild verifies that the PJRT C API header is present.
*/
func (config PJRTConfig) ValidateBuild() error {
	if config.IncludeDir == "" {
		return fmt.Errorf(
			"xla: missing PJRT include directory; set compute.xla.include_dir in cmd/asset/config.yml",
		)
	}

	headerFile := config.HeaderFile()

	if !pjrtFileExists(headerFile) {
		return fmt.Errorf("xla: PJRT header not found at %s", headerFile)
	}

	return nil
}

/*
ValidateRuntime verifies that the requested PJRT plugin is discoverable.
*/
func (config PJRTConfig) ValidateRuntime() error {
	resolvedPluginFile := config.ResolvedPluginFile()

	if resolvedPluginFile != "" {
		return nil
	}

	if config.PluginFile != "" {
		return fmt.Errorf("xla: PJRT plugin not found at %s", config.PluginFile)
	}

	return fmt.Errorf(
		"xla: %s not found; set compute.xla.%s_plugin_file, compute.xla.shared_plugin_file, or compute.xla.library_dirs in cmd/asset/config.yml",
		config.PluginName(), config.Platform,
	)
}

/*
Validate verifies the complete PJRT build and runtime environment.
*/
func (config PJRTConfig) Validate() error {
	if err := config.ValidateBuild(); err != nil {
		return err
	}

	return config.ValidateRuntime()
}

/*
normalizedPJRTPlatform maps user input to "cpu", "gpu", or "tpu".
Allowed inputs: "", whitespace (cpu default), cpu, gpu, cuda (cuda → gpu), tpu.
Any other token returns an error so typos are not silently treated as CPU.
*/
func normalizedPJRTPlatform(platform string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(platform))

	if normalized == "" {
		return "cpu", nil
	}

	switch normalized {
	case "cpu":
		return "cpu", nil
	case "gpu", "cuda":
		return "gpu", nil
	case "tpu":
		return "tpu", nil
	default:
		return "", fmt.Errorf(
			"xla: unsupported PJRT platform %q (allowed: cpu, gpu, cuda, tpu)",
			platform,
		)
	}
}

func pjrtFileExists(path string) bool {
	fileInfo, err := os.Stat(path)

	return err == nil && !fileInfo.IsDir()
}
