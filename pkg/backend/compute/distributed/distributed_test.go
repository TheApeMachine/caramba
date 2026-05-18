package distributed

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMeshRank_RoundTrip(t *testing.T) {
	convey.Convey("Given a 4x2 mesh", t, func() {
		mesh := tensor.ShardingMesh{
			Shape: []int{4, 2},
		}

		convey.Convey("Rank should agree with C-order traversal", func() {
			rank, err := MeshRank(mesh, []int{0, 0})
			convey.So(err, convey.ShouldBeNil)
			convey.So(rank, convey.ShouldEqual, 0)

			rank, err = MeshRank(mesh, []int{0, 1})
			convey.So(err, convey.ShouldBeNil)
			convey.So(rank, convey.ShouldEqual, 1)

			rank, err = MeshRank(mesh, []int{1, 0})
			convey.So(err, convey.ShouldBeNil)
			convey.So(rank, convey.ShouldEqual, 2)

			rank, err = MeshRank(mesh, []int{3, 1})
			convey.So(err, convey.ShouldBeNil)
			convey.So(rank, convey.ShouldEqual, 7)
		})

		convey.Convey("MeshCoords should invert MeshRank", func() {
			for inputRank := 0; inputRank < 8; inputRank++ {
				coords, err := MeshCoords(mesh, inputRank)
				convey.So(err, convey.ShouldBeNil)

				roundTripRank, err := MeshRank(mesh, coords)
				convey.So(err, convey.ShouldBeNil)
				convey.So(roundTripRank, convey.ShouldEqual, inputRank)
			}
		})

		convey.Convey("Out-of-range coords should error", func() {
			_, err := MeshRank(mesh, []int{4, 0})
			convey.So(err, convey.ShouldEqual, tensor.ErrShapeMismatch)

			_, err = MeshRank(mesh, []int{0, 2})
			convey.So(err, convey.ShouldEqual, tensor.ErrShapeMismatch)
		})
	})
}
