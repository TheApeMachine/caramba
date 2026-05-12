package config

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestNewComputeConfig(test *testing.T) {
	Convey("Given XLA compute settings", test, func() {
		setComputeConfigValue(test, "compute.xla.include_dir", "/opt/xla")
		setComputeConfigValue(test, "compute.xla.cpu_plugin_file", "/opt/pjrt/cpu.so")
		setComputeConfigValue(test, "compute.xla.gpu_plugin_file", "/opt/pjrt/gpu.so")
		setComputeConfigValue(test, "compute.xla.shared_plugin_file", "/opt/pjrt/shared.so")
		setComputeConfigValue(test, "compute.xla.library_dirs", []string{"/opt/pjrt", "", " /usr/lib/pjrt "})

		computeConfig := NewComputeConfig()

		Convey("It should load XLA paths from config", func() {
			So(computeConfig.XLA.IncludeDir, ShouldEqual, "/opt/xla")
			So(computeConfig.XLA.PluginFile("cpu"), ShouldEqual, "/opt/pjrt/cpu.so")
			So(computeConfig.XLA.PluginFile("gpu"), ShouldEqual, "/opt/pjrt/gpu.so")
			So(computeConfig.XLA.PluginFile("tpu"), ShouldEqual, "/opt/pjrt/shared.so")
			So(computeConfig.XLA.LibraryDirs, ShouldResemble, []string{"/opt/pjrt", "/usr/lib/pjrt"})
		})
	})
}

func BenchmarkNewComputeConfig(benchmark *testing.B) {
	setComputeConfigValue(benchmark, "compute.xla.include_dir", "/opt/xla")

	for benchmark.Loop() {
		_ = NewComputeConfig()
	}
}

func setComputeConfigValue(testingHandle interface {
	Helper()
	Cleanup(func())
}, key string, value any) {
	testingHandle.Helper()

	viper.Set(key, value)
	testingHandle.Cleanup(func() {
		viper.Set(key, nil)
	})
}
