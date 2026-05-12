package config

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestNewErrnieConfig(t *testing.T) {
	Convey("Given NewErrnieConfig", t, func() {
		Convey("When viper has no keys it should use defaults", func() {
			viper.Reset()

			cfg := NewErrnieConfig()

			So(cfg.Level, ShouldEqual, "info")
			So(cfg.File.Active, ShouldBeFalse)
			So(cfg.File.Path, ShouldEqual, "")
			So(cfg.Elasticsearch.Active, ShouldBeFalse)
			So(cfg.Elasticsearch.URL, ShouldEqual, "")
			So(cfg.Elasticsearch.Index, ShouldEqual, "")
			So(cfg.Elasticsearch.Username, ShouldEqual, "")
			So(cfg.Elasticsearch.Password, ShouldEqual, "")
		})

		Convey("When viper sets overrides they should appear on the struct", func() {
			viper.Reset()

			viper.Set("errnie.level", "warn")
			viper.Set("errnie.file.active", true)
			viper.Set("errnie.file.path", "/tmp/errnie.log")
			viper.Set("errnie.elasticsearch.active", true)
			viper.Set("errnie.elasticsearch.url", "http://localhost:9200")
			viper.Set("errnie.elasticsearch.index", "caramba")
			viper.Set("errnie.elasticsearch.username", "u")
			viper.Set("errnie.elasticsearch.password", "p")

			cfg := NewErrnieConfig()

			So(cfg.Level, ShouldEqual, "warn")
			So(cfg.File.Active, ShouldBeTrue)
			So(cfg.File.Path, ShouldEqual, "/tmp/errnie.log")
			So(cfg.Elasticsearch.Active, ShouldBeTrue)
			So(cfg.Elasticsearch.URL, ShouldEqual, "http://localhost:9200")
			So(cfg.Elasticsearch.Index, ShouldEqual, "caramba")
			So(cfg.Elasticsearch.Username, ShouldEqual, "u")
			So(cfg.Elasticsearch.Password, ShouldEqual, "p")
		})
	})
}

func BenchmarkNewErrnieConfig(b *testing.B) {
	viper.Reset()

	for b.Loop() {
		_ = NewErrnieConfig()
	}
}
