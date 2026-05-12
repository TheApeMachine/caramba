package config

import (
	"path/filepath"
	"strings"
)

var computeRootKey = "compute"

type ComputeConfig struct {
	Metal MetalConfig
	XLA   XLAConfig
}

type MetalConfig struct {
	MetallibDirectory string
}

type XLAConfig struct {
	IncludeDir       string
	CPUPluginFile    string
	GPUPluginFile    string
	TPUPluginFile    string
	SharedPluginFile string
	LibraryDirs      []string
}

func NewComputeConfig() *ComputeConfig {
	return &ComputeConfig{
		Metal: MetalConfig{
			MetallibDirectory: WithDefault(
				computeRootKey+".metal.metallib_directory",
				".",
			),
		},
		XLA: XLAConfig{
			IncludeDir: WithDefault(computeRootKey+".xla.include_dir", ""),
			CPUPluginFile: WithDefault(
				computeRootKey+".xla.cpu_plugin_file",
				"",
			),
			GPUPluginFile: WithDefault(
				computeRootKey+".xla.gpu_plugin_file",
				"",
			),
			TPUPluginFile: WithDefault(
				computeRootKey+".xla.tpu_plugin_file",
				"",
			),
			SharedPluginFile: WithDefault(
				computeRootKey+".xla.shared_plugin_file",
				"",
			),
			LibraryDirs: cleanStringSlice(WithDefault(
				computeRootKey+".xla.library_dirs",
				[]string{},
			)),
		},
	}
}

func (metalConfig MetalConfig) Metallib(name string) string {
	return filepath.Join(metalConfig.MetallibDirectory, name)
}

func (xlaConfig XLAConfig) PluginFile(platform string) string {
	switch platform {
	case "gpu":
		if xlaConfig.GPUPluginFile != "" {
			return xlaConfig.GPUPluginFile
		}
	case "tpu":
		if xlaConfig.TPUPluginFile != "" {
			return xlaConfig.TPUPluginFile
		}
	default:
		if xlaConfig.CPUPluginFile != "" {
			return xlaConfig.CPUPluginFile
		}
	}

	return xlaConfig.SharedPluginFile
}

func cleanStringSlice(values []string) []string {
	cleaned := make([]string, 0, len(values))

	for _, value := range values {
		trimmed := strings.TrimSpace(value)

		if trimmed == "" {
			continue
		}

		cleaned = append(cleaned, trimmed)
	}

	return cleaned
}
