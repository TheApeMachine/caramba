package neo4j

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestConfigFromEnv(test *testing.T) {
	Convey("ConfigFromEnv reads store.neo4j settings", test, func() {
		setNeo4jConfigValue(test, "store.neo4j.uri", "neo4j://db:7687")
		setNeo4jConfigValue(test, "store.neo4j.username", "neo4j")
		setNeo4jConfigValue(test, "store.neo4j.password", "secret")

		cfg := ConfigFromEnv()
		So(cfg.URI, ShouldEqual, "neo4j://db:7687")
		So(cfg.Username, ShouldEqual, "neo4j")
		So(cfg.Password, ShouldEqual, "secret")
	})
}

func TestNewClient(test *testing.T) {
	Convey("NewClient", test, func() {
		Convey("rejects empty URI", func() {
			_, err := NewClient(Config{URI: ""})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "URI")
		})

		Convey("accepts URI string", func() {
			_, err := NewClient(Config{URI: "neo4j://localhost:7687", Username: "u", Password: "p"})
			So(err, ShouldBeNil)
		})
	})
}

func setNeo4jConfigValue(testingHandle interface {
	Helper()
	Cleanup(func())
}, key string, value any) {
	testingHandle.Helper()

	viper.Set(key, value)
	testingHandle.Cleanup(func() {
		viper.Set(key, nil)
	})
}
