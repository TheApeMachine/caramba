package qdrant

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestConfigFromEnv(test *testing.T) {
	Convey("ConfigFromEnv", test, func() {
		setQdrantConfigValue(test, "store.qdrant.host", "")
		setQdrantConfigValue(test, "store.qdrant.grpc_port", 0)
		setQdrantConfigValue(test, "store.qdrant.port", 0)
		setQdrantConfigValue(test, "store.qdrant.url", "http://example:6333")
		setQdrantConfigValue(test, "store.qdrant.use_tls", false)
		setQdrantConfigValue(test, "store.qdrant.api_key", "")

		cfg := ConfigFromEnv()
		So(cfg.Host, ShouldEqual, "example")
		So(cfg.Port, ShouldEqual, 6334)
		So(cfg.UseTLS, ShouldBeFalse)
	})

	Convey("HTTPS URL enables TLS", test, func() {
		setQdrantConfigValue(test, "store.qdrant.url", "https://example:6334")
		setQdrantConfigValue(test, "store.qdrant.grpc_port", 0)
		setQdrantConfigValue(test, "store.qdrant.port", 0)
		setQdrantConfigValue(test, "store.qdrant.use_tls", false)

		cfg := ConfigFromEnv()
		So(cfg.UseTLS, ShouldBeTrue)
		So(cfg.Host, ShouldEqual, "example")
	})

	Convey("explicit gRPC port wins over URL", test, func() {
		setQdrantConfigValue(test, "store.qdrant.url", "http://h:6333")
		setQdrantConfigValue(test, "store.qdrant.grpc_port", 7777)
		setQdrantConfigValue(test, "store.qdrant.port", 0)

		cfg := ConfigFromEnv()
		So(cfg.Port, ShouldEqual, 7777)
	})

	Convey("default host and port without config", test, func() {
		viper.Reset()
		test.Cleanup(viper.Reset)

		cfg := ConfigFromEnv()
		So(cfg.Host, ShouldEqual, defaultGRPCHost)
		So(cfg.Port, ShouldEqual, defaultGRPCPort)
	})
}

func TestNewClient(test *testing.T) {
	Convey("NewClient uses defaults for host and port", test, func() {
		client, err := NewClient(Config{})
		So(err, ShouldBeNil)
		So(client, ShouldNotBeNil)
		So(client.Native(), ShouldNotBeNil)
		So(client.Close(), ShouldBeNil)
	})
}

func TestMergeURLOverrides_preservesExplicitGRPCPortFromURL(test *testing.T) {
	Convey("non-6333 URL port is kept", test, func() {
		host, port, useTLS := mergeURLOverrides("http://x:9999", "", 0, false)
		So(host, ShouldEqual, "x")
		So(port, ShouldEqual, 9999)
		So(useTLS, ShouldBeFalse)
	})
}

func setQdrantConfigValue(testingHandle interface {
	Helper()
	Cleanup(func())
}, key string, value any) {
	testingHandle.Helper()

	viper.Set(key, value)
	testingHandle.Cleanup(func() {
		viper.Set(key, nil)
	})
}
