package xla

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
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
NewPJRTConfig reads the environment used by the XLA backend.
*/
func NewPJRTConfig(platform string) PJRTConfig {
	return PJRTConfig{
		Platform:    normalizedPJRTPlatform(platform),
		IncludeDir:  pjrtIncludeDir(),
		PluginFile:  pjrtPluginFile(platform),
		LibraryDirs: pjrtLibraryDirs(),
	}
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
			"xla: missing PJRT include directory; set CARAMBA_XLA_INCLUDE_DIR or CGO_CPPFLAGS=-I/path/to/xla",
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
		"xla: %s not found; set CARAMBA_PJRT_%s_PLUGIN, CARAMBA_PJRT_PLUGIN, or CARAMBA_PJRT_LIBRARY_DIR",
		config.PluginName(), strings.ToUpper(config.Platform),
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

func normalizedPJRTPlatform(platform string) string {
	normalized := strings.ToLower(strings.TrimSpace(platform))

	if normalized == "gpu" || normalized == "cuda" {
		return "gpu"
	}

	return "cpu"
}

func pjrtIncludeDir() string {
	for _, envName := range []string{"CARAMBA_XLA_INCLUDE_DIR", "XLA_INCLUDE_DIR", "XLA_INCLUDE"} {
		if value := strings.TrimSpace(os.Getenv(envName)); value != "" {
			return value
		}
	}

	return pjrtIncludeDirFromCGOFlags(os.Getenv("CGO_CPPFLAGS"))
}

func pjrtIncludeDirFromCGOFlags(cgoFlags string) string {
	fields := strings.Fields(cgoFlags)

	for fieldIndex, field := range fields {
		if strings.HasPrefix(field, "-I") && len(field) > 2 {
			return strings.TrimPrefix(field, "-I")
		}

		if field == "-I" && fieldIndex+1 < len(fields) {
			return fields[fieldIndex+1]
		}
	}

	return ""
}

func pjrtPluginFile(platform string) string {
	platformEnvName := "CARAMBA_PJRT_CPU_PLUGIN"

	if normalizedPJRTPlatform(platform) == "gpu" {
		platformEnvName = "CARAMBA_PJRT_GPU_PLUGIN"
	}

	for _, envName := range []string{platformEnvName, "CARAMBA_PJRT_PLUGIN", "PJRT_PLUGIN_PATH"} {
		if value := strings.TrimSpace(os.Getenv(envName)); value != "" {
			return value
		}
	}

	return ""
}

func pjrtLibraryDirs() []string {
	var libraryDirs []string

	for _, envName := range []string{
		"CARAMBA_PJRT_LIBRARY_DIR",
		"LD_LIBRARY_PATH",
		"DYLD_LIBRARY_PATH",
	} {
		for _, libraryDir := range filepath.SplitList(os.Getenv(envName)) {
			if libraryDir == "" || slices.Contains(libraryDirs, libraryDir) {
				continue
			}

			libraryDirs = append(libraryDirs, libraryDir)
		}
	}

	return libraryDirs
}

func pjrtFileExists(path string) bool {
	fileInfo, err := os.Stat(path)

	return err == nil && !fileInfo.IsDir()
}
