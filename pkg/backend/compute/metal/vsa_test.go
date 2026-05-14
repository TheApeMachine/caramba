//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpuvsa "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/vsa"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestMetalVSAOps(t *testing.T) {
	lib := metallibPathOrSkip(t, "vsa.metallib")

	Convey("Given initialized MetalVSAOps", t, func() {
		vsaOps, err := NewVSAOps(lib)
		So(err, ShouldBeNil)
		defer func() {
			So(vsaOps.Close(), ShouldBeNil)
		}()

		Convey("It should bundle vectors on Metal", func() {
			left := []float64{1, 0, 0, 0}
			right := []float64{0, 1, 0, 0}
			expectedState, err := cpuvsa.NewBundle().Forward(
				state.NewDict().WithShape([]int{4}).WithInputs(left, right),
			)

			So(err, ShouldBeNil)
			output, err := vsaOps.Bundle([]int{4}, left, right)

			So(err, ShouldBeNil)
			for index := range output {
				So(math.Abs(output[index]-expectedState.Out[index]) < 1e-5, ShouldBeTrue)
			}
		})

		Convey("It should permute and inverse-permute on Metal", func() {
			input := []float64{1, 2, 3, 4}
			stateDict := state.NewDict().WithShape([]int{4}).WithInput(input)
			stateDict.K = 1
			expectedState, err := cpuvsa.NewPermute().Forward(stateDict)

			So(err, ShouldBeNil)
			output, err := vsaOps.Permute([]int{4}, 1, input)

			So(err, ShouldBeNil)
			So(output, ShouldResemble, expectedState.Out)

			recovered, err := vsaOps.InversePermute([]int{4}, 1, output)

			So(err, ShouldBeNil)
			So(recovered, ShouldResemble, input)
		})
	})
}

func BenchmarkMetalVSAOpsPermute(b *testing.B) {
	vsaOps, err := NewVSAOps(testdataPathMetalLib("vsa.metallib"))

	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		_ = vsaOps.Close()
	}()

	input := []float64{1, 2, 3, 4}

	for b.Loop() {
		_, _ = vsaOps.Permute([]int{4}, 1, input)
	}
}
