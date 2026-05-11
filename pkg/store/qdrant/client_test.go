package qdrant

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestConfigFromEnv(t *testing.T) {
	Convey("ConfigFromEnv", t, func() {
		t.Setenv("QDRANT_HOST", "")
		t.Setenv("QDRANT_GRPC_PORT", "")
		t.Setenv("QDRANT_PORT", "")
		t.Setenv("QDRANT_URL", "http://example:6333")
		t.Setenv("QDRANT_USE_TLS", "")
		t.Setenv("QDRANT_API_KEY", "")

		cfg := ConfigFromEnv()
		So(cfg.Host, ShouldEqual, "example")
		So(cfg.Port, ShouldEqual, 6334)
		So(cfg.UseTLS, ShouldBeFalse)
	})

	Convey("HTTPS URL enables TLS", t, func() {
		t.Setenv("QDRANT_URL", "https://example:6334")
		t.Setenv("QDRANT_GRPC_PORT", "")
		t.Setenv("QDRANT_PORT", "")
		t.Setenv("QDRANT_USE_TLS", "")

		cfg := ConfigFromEnv()
		So(cfg.UseTLS, ShouldBeTrue)
		So(cfg.Host, ShouldEqual, "example")
	})

	Convey("explicit gRPC port wins over URL", t, func() {
		t.Setenv("QDRANT_URL", "http://h:6333")
		t.Setenv("QDRANT_GRPC_PORT", "7777")
		t.Setenv("QDRANT_PORT", "")

		cfg := ConfigFromEnv()
		So(cfg.Port, ShouldEqual, 7777)
	})

	Convey("default host and port without env", t, func() {
		for _, k := range []string{
			"QDRANT_URL", "QDRANT_BASE_URL", "QDRANT_HOST", "QDRANT_GRPC_PORT", "QDRANT_PORT",
		} {
			t.Setenv(k, "")
		}

		cfg := ConfigFromEnv()
		So(cfg.Host, ShouldEqual, defaultGRPCHost)
		So(cfg.Port, ShouldEqual, defaultGRPCPort)
	})
}

func TestNewClient(t *testing.T) {
	Convey("NewClient uses defaults for host and port", t, func() {
		c, err := NewClient(Config{})
		So(err, ShouldBeNil)
		So(c, ShouldNotBeNil)
		So(c.Native(), ShouldNotBeNil)
		So(c.Close(), ShouldBeNil)
	})
}

func TestMergeURLOverrides_preservesExplicitGRPCPortFromURL(t *testing.T) {
	Convey("non-6333 URL port is kept", t, func() {
		h, p, tls := mergeURLOverrides("http://x:9999", "", 0, false)
		So(h, ShouldEqual, "x")
		So(p, ShouldEqual, 9999)
		So(tls, ShouldBeFalse)
	})
}
