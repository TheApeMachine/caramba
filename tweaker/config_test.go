package tweaker

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func setupTestViper() {
	viper.Reset()
	viper.Set("test.key1", "value1")
	viper.Set("test.key2", "value2")
	viper.Set("test.key3", "value3")
}

func TestConfig(t *testing.T) {
	Convey("Given a new Config", t, func() {
		setupTestViper()
		cfg := NewConfig("test")

		Convey("It should initialize correctly", func() {
			So(cfg, ShouldNotBeNil)
			So(cfg.baseKey, ShouldEqual, "test")
		})

		Convey("When S is called with a single key", func() {
			value := cfg.S("key1")
			So(value, ShouldEqual, "value1")
		})

		Convey("When S is called with multiple keys", func() {
			Convey("It should round-robin through the values", func() {
				So(cfg.S("key1", "key2", "key3"), ShouldEqual, "value1")
				So(cfg.S("key1", "key2", "key3"), ShouldEqual, "value2")
				So(cfg.S("key1", "key2", "key3"), ShouldEqual, "value3")
				So(cfg.S("key1", "key2", "key3"), ShouldEqual, "value1") // Back to start
			})
		})

		Convey("When S is called with non-existent keys", func() {
			So(cfg.S("nonexistent"), ShouldEqual, "")
		})

		Convey("When S is called with no keys", func() {
			So(cfg.S(), ShouldEqual, "")
		})

		Convey("Given a previously cached value", func() {
			cfg.S("key1")                            // Cache the value
			viper.Set("test.key1", "changed")        // Change won't affect cached value
			So(cfg.S("key1"), ShouldEqual, "value1") // Should use cached value
		})
	})
}
