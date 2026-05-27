package weights

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestStoreLookupTransposedSliceReadsCheckpointBytes(testingObject *testing.T) {
	convey.Convey("Given a safetensors store with a packed projection tensor", testingObject, func() {
		archivePath := writeSafetensorsFixture(
			testingObject,
			"packed.weight",
			[]int{5, 3},
			[]float32{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				10, 11, 12,
				13, 14, 15,
			},
		)
		memory := &recordingWeightBackend{}

		store, err := New(memory, []string{archivePath})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should slice output rows before transposing", func() {
			resident, err := store.LookupTransposedSlice("packed.weight", "output", 1, 4)

			convey.So(err, convey.ShouldBeNil)
			convey.So(resident, convey.ShouldNotBeNil)
			convey.So(memory.uploads, convey.ShouldHaveLength, 1)
			convey.So(memory.uploads[0].shape.Dims(), convey.ShouldResemble, []int{3, 3})
			convey.So(float32BytesToValues(memory.uploads[0].payload), convey.ShouldResemble, []float32{
				4, 7, 10,
				5, 8, 11,
				6, 9, 12,
			})
		})
	})
}

func TestStoreLookupTransposedSliceReadsInputAxisCheckpointBytes(testingObject *testing.T) {
	convey.Convey("Given a safetensors store with a wide projection tensor", testingObject, func() {
		archivePath := writeSafetensorsFixture(
			testingObject,
			"wide.weight",
			[]int{3, 5},
			[]float32{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 10,
				11, 12, 13, 14, 15,
			},
		)
		memory := &recordingWeightBackend{}

		store, err := New(memory, []string{archivePath})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should slice input columns before transposing", func() {
			resident, err := store.LookupTransposedSlice("wide.weight", "input", 1, 4)

			convey.So(err, convey.ShouldBeNil)
			convey.So(resident, convey.ShouldNotBeNil)
			convey.So(memory.uploads, convey.ShouldHaveLength, 1)
			convey.So(memory.uploads[0].shape.Dims(), convey.ShouldResemble, []int{3, 3})
			convey.So(float32BytesToValues(memory.uploads[0].payload), convey.ShouldResemble, []float32{
				2, 7, 12,
				3, 8, 13,
				4, 9, 14,
			})
		})
	})
}
