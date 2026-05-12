package errnie

import (
	"testing"

	"github.com/phuslu/log"
	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/config"
)

func TestApply(t *testing.T) {
	Convey("Given Apply", t, func() {
		saved := log.DefaultLogger
		defer func() {
			log.DefaultLogger = saved
		}()

		Convey("When level is debug", func() {
			Apply(&config.ErrnieConfig{Level: "debug"})
			So(log.DefaultLogger.Level, ShouldEqual, log.DebugLevel)
		})

		Convey("When level is unknown it should default to info", func() {
			Apply(&config.ErrnieConfig{Level: "not-a-level"})
			So(log.DefaultLogger.Level, ShouldEqual, log.InfoLevel)
		})
	})
}

func BenchmarkApply(b *testing.B) {
	cfg := &config.ErrnieConfig{Level: "info"}

	for range b.N {
		Apply(cfg)
	}
}
