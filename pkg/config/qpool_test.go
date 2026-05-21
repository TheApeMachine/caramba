package config

import (
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewQPoolConfig(test *testing.T) {
	Convey("Given qpool settings in config", test, func() {
		setComputeConfigValue(test, "qpool.min_workers", 4)
		setComputeConfigValue(test, "qpool.max_workers", 16)
		setComputeConfigValue(test, "qpool.scheduling_timeout", 5*time.Second)
		setComputeConfigValue(test, "qpool.job_channel_capacity", 128)

		qpoolConfig := NewQPoolConfig()

		Convey("It should load worker pool settings from config", func() {
			So(qpoolConfig.MinWorkers, ShouldEqual, 4)
			So(qpoolConfig.MaxWorkers, ShouldEqual, 16)
			So(qpoolConfig.SchedulingTimeout, ShouldEqual, 5*time.Second)
			So(qpoolConfig.JobChannelCapacity, ShouldEqual, 128)
		})
	})
}

func BenchmarkNewQPoolConfig(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = NewQPoolConfig()
	}
}
