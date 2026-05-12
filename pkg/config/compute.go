package config

import "path/filepath"

var computeRootKey = "compute"

type ComputeConfig struct {
	Metal MetalConfig
}

type MetalConfig struct {
	MetallibDirectory string
}

func NewComputeConfig() *ComputeConfig {
	return &ComputeConfig{
		Metal: MetalConfig{
			MetallibDirectory: WithDefault(
				computeRootKey+".metal.metallib_directory",
				".",
			),
		},
	}
}

func (metalConfig MetalConfig) Metallib(name string) string {
	return filepath.Join(metalConfig.MetallibDirectory, name)
}
