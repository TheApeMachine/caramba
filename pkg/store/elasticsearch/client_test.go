package elasticsearch

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestNewClient(test *testing.T) {
	Convey("NewClient", test, func() {
		Convey("rejects no addresses", func() {
			_, err := NewClient(Config{Addresses: nil})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "address")
		})

		Convey("rejects all blank addresses", func() {
			_, err := NewClient(Config{Addresses: []string{" ", ""}})
			So(err, ShouldNotBeNil)
		})

		Convey("accepts trimmed addresses", func() {
			_, err := NewClient(Config{Addresses: []string{" http://localhost:9200 "}})
			So(err, ShouldBeNil)
		})
	})
}

func TestConfigFromEnv(test *testing.T) {
	Convey("ConfigFromEnv reads store.elasticsearch settings", test, func() {
		setElasticsearchConfigValue(test, "store.elasticsearch.addresses", "http://a:9200,http://b:9200")
		setElasticsearchConfigValue(test, "store.elasticsearch.url", "")
		setElasticsearchConfigValue(test, "store.elasticsearch.username", "elastic")
		setElasticsearchConfigValue(test, "store.elasticsearch.api_key", "es-key")

		cfg := ConfigFromEnv()
		So(cfg.Addresses, ShouldResemble, []string{"http://a:9200", "http://b:9200"})
		So(cfg.Username, ShouldEqual, "elastic")
		So(cfg.APIKey, ShouldEqual, "es-key")
	})

	Convey("URL is used when addresses are empty", test, func() {
		setElasticsearchConfigValue(test, "store.elasticsearch.addresses", "")
		setElasticsearchConfigValue(test, "store.elasticsearch.url", "http://localhost:9200")

		cfg := ConfigFromEnv()
		So(cfg.Addresses, ShouldResemble, []string{"http://localhost:9200"})
	})
}

func setElasticsearchConfigValue(testingHandle interface {
	Helper()
	Cleanup(func())
}, key string, value any) {
	testingHandle.Helper()

	viper.Set(key, value)
	testingHandle.Cleanup(func() {
		viper.Set(key, nil)
	})
}
