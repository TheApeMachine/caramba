package cmd

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/tensor"
)

func TestComputeDeviceLocation(testingObject *testing.T) {
	convey.Convey("Given configured compute device names", testingObject, func() {
		cases := map[string]tensor.Location{
			"host":  tensor.Host,
			"cpu":   tensor.Host,
			"metal": tensor.Metal,
			"cuda":  tensor.CUDA,
			"xla":   tensor.XLA,
		}

		convey.Convey("It should resolve supported locations", func() {
			for raw, expected := range cases {
				location, err := computeDeviceLocation(raw)

				convey.So(err, convey.ShouldBeNil)
				convey.So(location, convey.ShouldEqual, expected)
			}
		})

		convey.Convey("It should reject unknown locations", func() {
			_, err := computeDeviceLocation("quantum")

			convey.So(err, convey.ShouldNotBeNil)
		})
	})
}
