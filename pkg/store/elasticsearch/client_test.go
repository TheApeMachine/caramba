package elasticsearch

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewClient(t *testing.T) {
	Convey("NewClient", t, func() {
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
			// Host may be unreachable; client construction should succeed.
			So(err, ShouldBeNil)
		})
	})
}
