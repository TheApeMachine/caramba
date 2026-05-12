//go:build darwin && cgo

package metal

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	cpuai "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/active_inference"
)

func metallibPathOrSkip(t *testing.T, name string) string {
	t.Helper()

	_, file, _, ok := runtime.Caller(0)

	if !ok {
		t.Skip("runtime.Caller failed")
	}

	p := filepath.Join(filepath.Dir(file), name)

	if _, err := os.Stat(p); err != nil {
		t.Skipf("missing %s — run `make build` in repo root", p)
	}

	return p
}

func TestActiveInferenceOps_FreeEnergy_ParityWithCPU(t *testing.T) {
	lib := metallibPathOrSkip(t, "active_inference.metallib")

	Convey("Metal free energy matches CPU reference on float32-sized data", t, func() {
		ops, err := NewActiveInferenceOps(lib)

		So(err, ShouldBeNil)
		defer func() {
			So(ops.Close(), ShouldBeNil)
		}()

		n := 256
		mu := make([]float64, n)
		ls := make([]float64, n)

		for i := 0; i < n; i++ {
			mu[i] = 0.01 * float64(i%17-8)
			ls[i] = -0.5 + 0.03*float64(i%13)
		}

		cpuOut := cpuai.NewFreeEnergy().Forward([]int{n}, mu, ls)
		mo, errM := ops.FreeEnergy([]int{n}, mu, ls)

		So(errM, ShouldBeNil)
		So(len(mo), ShouldEqual, 1)
		So(len(cpuOut), ShouldEqual, 1)
		So(math.Abs(mo[0]-cpuOut[0]) < 5e-3, ShouldBeTrue)
	})
}
