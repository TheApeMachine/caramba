package weights

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestTensorBindingFromMetadata(test *testing.T) {
	Convey("Given from_safetensors node metadata", test, func() {
		metadata := map[string]any{
			"from_safetensors": map[string]any{
				"weight":      "packed.weight",
				"bias":        "packed.bias",
				"slice_axis":  "output",
				"slice_start": 4,
			},
		}

		Convey("It should decode direct tensor binding fields", func() {
			binding, handled, err := tensorBindingFromMetadata(metadata)

			So(err, ShouldBeNil)
			So(handled, ShouldBeTrue)
			So(binding.weightName, ShouldEqual, "packed.weight")
			So(binding.biasName, ShouldEqual, "packed.bias")
			So(binding.sliceAxis, ShouldEqual, "output")
			So(binding.sliceStart, ShouldEqual, 4)
		})
	})
}
