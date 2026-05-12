package config

import (
	"fmt"
	"os"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestWithDefault(t *testing.T) {
	Convey("Given WithDefault", t, func() {
		unique := fmt.Sprintf("config.test.%d", time.Now().UnixNano())

		Convey("When the value is a string containing ${VAR}", func() {
			Convey("It should expand using the environment", func() {
				key := unique + ".expand"
				setErr := os.Setenv("CARAMBA_WITHDEFAULT_TEST", "xy")
				So(setErr, ShouldBeNil)
				defer os.Unsetenv("CARAMBA_WITHDEFAULT_TEST")

				viper.Set(key, "a-${CARAMBA_WITHDEFAULT_TEST}-b")
				defer viper.Set(key, nil)

				So(WithDefault(key, "unused"), ShouldEqual, "a-xy-b")
			})

			Convey("It should expand the default when the key is not set", func() {
				missingKey := unique + ".missing"
				setErr := os.Setenv("CARAMBA_WITHDEFAULT_DEF", "z")
				So(setErr, ShouldBeNil)
				defer os.Unsetenv("CARAMBA_WITHDEFAULT_DEF")

				So(WithDefault(missingKey, "q-${CARAMBA_WITHDEFAULT_DEF}-r"), ShouldEqual, "q-z-r")
			})
		})

		Convey("When the value is bool it should not alter it", func() {
			key := unique + ".bool"
			viper.Set(key, true)
			defer viper.Set(key, nil)

			So(WithDefault(key, false), ShouldBeTrue)
		})
	})
}

func BenchmarkWithDefault(b *testing.B) {
	key := fmt.Sprintf("config.bench.%d", time.Now().UnixNano())
	viper.Set(key, "static")
	b.ResetTimer()

	for range b.N {
		_ = WithDefault(key, "default")
	}
}
