package config

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestNewHubConfig(test *testing.T) {
	Convey("Given Hub settings", test, func() {
		setHubConfigValue(test, "hub.endpoint", "https://hf.example")
		setHubConfigValue(test, "hub.cache_dir", "/tmp/caramba-hub")
		setHubConfigValue(test, "hub.token", "hf_test")
		setHubConfigValue(test, "hub.offline", true)
		setHubConfigValue(test, "hub.max_workers", 3)
		setHubConfigValue(test, "hub.xet.active", false)

		hubConfig := NewHubConfig()

		Convey("It should load Hub settings from config", func() {
			So(hubConfig.Endpoint, ShouldEqual, "https://hf.example")
			So(hubConfig.CacheDir, ShouldEqual, "/tmp/caramba-hub")
			So(hubConfig.Token, ShouldEqual, "hf_test")
			So(hubConfig.Offline, ShouldBeTrue)
			So(hubConfig.MaxWorkers, ShouldEqual, 3)
			So(hubConfig.Xet.Active, ShouldBeFalse)
		})
	})
}

func BenchmarkNewHubConfig(benchmark *testing.B) {
	setHubConfigValue(benchmark, "hub.cache_dir", "/tmp/caramba-hub")

	for benchmark.Loop() {
		_ = NewHubConfig()
	}
}

func setHubConfigValue(testingHandle interface {
	Helper()
	Cleanup(func())
}, key string, value any) {
	testingHandle.Helper()

	viper.Set(key, value)
	testingHandle.Cleanup(func() {
		viper.Set(key, nil)
	})
}
