package fusion

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestCatalog_Lookup_MatMulBiasGELU(t *testing.T) {
	convey.Convey("Given the default catalog", t, func() {
		convey.Convey("matmul+add+gelu in bf16 on host should match", func() {
			entry := Default.Lookup(
				[]string{"matmul", "add", "gelu"},
				[]dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.BFloat16},
				tensor.LayoutDense,
				tensor.Host,
			)

			convey.So(entry, convey.ShouldNotBeNil)
			convey.So(entry.FusedOp, convey.ShouldEqual, "matmul_bias_gelu")
			convey.So(entry.OutputDType, convey.ShouldEqual, dtype.BFloat16)
			convey.So(entry.ParityULPBound, convey.ShouldBeLessThanOrEqualTo, 2)
		})

		convey.Convey("Mixed-dtype matmul+add+gelu should not match the bf16 entry", func() {
			entry := Default.Lookup(
				[]string{"matmul", "add", "gelu"},
				[]dtype.DType{dtype.BFloat16, dtype.Float32, dtype.BFloat16},
				tensor.LayoutDense,
				tensor.Host,
			)

			convey.So(entry, convey.ShouldBeNil)
		})

		convey.Convey("XLA backend without a registered entry should miss", func() {
			entry := Default.Lookup(
				[]string{"matmul", "add", "gelu"},
				[]dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.BFloat16},
				tensor.LayoutDense,
				tensor.XLA,
			)

			convey.So(entry, convey.ShouldBeNil)
		})
	})
}

func TestCatalog_Entries(t *testing.T) {
	convey.Convey("Given the default catalog", t, func() {
		entries := Default.Entries()

		convey.Convey("It should contain at least the standard transformer fusions", func() {
			convey.So(len(entries), convey.ShouldBeGreaterThanOrEqualTo, 4)
		})
	})
}
