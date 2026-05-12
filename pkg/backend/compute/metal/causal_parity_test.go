//go:build darwin && cgo

package metal

import (
	"math"
	"path/filepath"
	"runtime"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	cpucausal "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/causal"
)

func testdataPathMetalLib(name string) string {
	_, file, _, ok := runtime.Caller(0)

	if !ok {
		return name
	}

	dir := filepath.Dir(file)

	return filepath.Join(dir, name)
}

func TestMetalCausalOps_DoCalculus_ParityWithCPU(t *testing.T) {
	Convey("Given identical Pearl do-calculus inputs", t, func() {
		dummyPath := testdataPathMetalLib("causal.metallib")
		metalOps, err := NewCausalOps(dummyPath)

		So(err, ShouldBeNil)
		defer func() {
			So(metalOps.Close(), ShouldBeNil)
		}()

		n := 3
		cov := []float64{
			1, 0.2, 0.1,
			0.2, 1, 0.3,
			0.1, 0.3, 1,
		}
		mask := []float64{0, 1, 0}
		values := []float64{0, 0.75, 0}

		cpu := cpucausal.NewDoCalculus()
		cpuOut := cpu.Forward([]int{n, n}, cov, mask, values)

		metaOut, errMeta := metalOps.DoCalculus([]int{n}, cov, mask, values)

		So(errMeta, ShouldBeNil)
		So(len(metaOut), ShouldEqual, len(cpuOut))

		for idx := range cpuOut {
			So(math.Abs(metaOut[idx]-cpuOut[idx]) < 1e-4, ShouldBeTrue)
		}
	})
}
