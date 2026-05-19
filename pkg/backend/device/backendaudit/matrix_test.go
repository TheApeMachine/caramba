package backendaudit

import (
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/device"
	"github.com/theapemachine/caramba/pkg/dtype"
	. "github.com/smartystreets/goconvey/convey"
)

func TestBuildDeviceBackendMatrix(t *testing.T) {
	Convey("Given Metal, CUDA, and XLA device backends", t, func() {
		matrix, err := BuildDeviceBackendMatrix()

		So(err, ShouldBeNil)
		So(matrix, ShouldNotBeNil)

		metalRow := backendRow(matrix, DeviceBackendMetal)
		cudaRow := backendRow(matrix, DeviceBackendCUDA)
		xlaRow := backendRow(matrix, DeviceBackendXLA)

		Convey("It should report Metal kernel registrations", func() {
			So(metalRow.RegisteredKernels, ShouldEqual, 462)
			So(metalRow.UniqueKernelNames, ShouldEqual, 158)
			So(metalRow.KernelSources, ShouldEqual, 24)
		})

		Convey("It should report Metal required-op kernel coverage", func() {
			metalYes := 0

			for _, row := range matrix.RequiredOps {
				if row.MetalKernels == KernelRegistered {
					metalYes++
				}
			}

			So(metalYes, ShouldEqual, 68)
		})

		Convey("It should report no CUDA or XLA kernel registrations", func() {
			So(cudaRow.RegisteredKernels, ShouldEqual, 0)
			So(xlaRow.RegisteredKernels, ShouldEqual, 0)
		})

		Convey("It should list Metal supported dtypes from backend.go", func() {
			So(metalRow.SupportedDTypes, ShouldResemble, []dtype.DType{
				dtype.Float32,
				dtype.BFloat16,
				dtype.Float16,
				dtype.Int32,
				dtype.Int8,
				dtype.Int4,
				dtype.Bool,
			})
		})

		Convey("It should list CUDA dtypes from bridge_real.go", func() {
			So(cudaRow.SupportedDTypes, ShouldContain, dtype.Float32)
			So(cudaRow.SupportedDTypes, ShouldContain, dtype.BFloat16)
			So(cudaRow.SupportedDTypes, ShouldContain, dtype.Float16)
			So(len(cudaRow.SupportedDTypes), ShouldBeGreaterThanOrEqualTo, 6)
		})

		Convey("It should list XLA supported dtypes from backend.go", func() {
			So(len(xlaRow.SupportedDTypes), ShouldEqual, 15)
		})

		Convey("It should register matmul and add on Metal only", func() {
			addRow := kernelRow(matrix, "add")
			matmulRow := kernelRow(matrix, "matmul")

			So(addRow.Metal, ShouldEqual, KernelRegistered)
			So(addRow.CUDA, ShouldEqual, KernelNotRegistered)
			So(matmulRow.Metal, ShouldEqual, KernelRegistered)
		})

		Convey("It should cover required ops with cross-link index size", func() {
			So(len(matrix.RequiredOps), ShouldEqual, len(ir.RequiredOperationIDs()))
		})

		Convey("It should register Metal kernels for core math ops", func() {
			So(requiredOpMetal(matrix, ir.OpAdd), ShouldEqual, KernelRegistered)
			So(requiredOpMetal(matrix, "math.add"), ShouldEqual, KernelRegistered)
			So(requiredOpMetal(matrix, "math.matmul"), ShouldEqual, KernelRegistered)
			So(requiredOpMetal(matrix, "positional.rope"), ShouldEqual, KernelRegistered)
		})
	})
}

func TestValidateDeviceBackendMatrix(t *testing.T) {
	Convey("Given the device backend matrix", t, func() {
		matrix, err := BuildDeviceBackendMatrix()

		So(err, ShouldBeNil)

		Convey("It should validate structural invariants", func() {
			So(ValidateDeviceBackendMatrix(matrix), ShouldBeNil)
		})
	})
}

func TestRenderMarkdown(t *testing.T) {
	Convey("Given the device backend matrix", t, func() {
		matrix, err := BuildDeviceBackendMatrix()

		So(err, ShouldBeNil)

		Convey("It should render backend summary headers", func() {
			doc := RenderMarkdown(matrix)

			So(doc, ShouldContainSubstring, "# Device backend matrix (T1.4)")
			So(doc, ShouldContainSubstring, "| Backend | Supported dtypes |")
			So(doc, ShouldContainSubstring, "| metal |")
		})
	})
}

func TestKernelNamesForOperation(t *testing.T) {
	Convey("Given operation cross-links", t, func() {
		index := device.BuildOperationCrossLinkIndex()

		Convey("It should map train.optimizer.adam to adam_step", func() {
			names := kernelNamesForOperation("train.optimizer.adam", index["train.optimizer.adam"])

			So(names, ShouldResemble, []string{"adam_step"})
		})

		Convey("It should map OpAdd to add via direct cross-link", func() {
			names := kernelNamesForOperation(ir.OpAdd, index[ir.OpAdd])

			So(names, ShouldResemble, []string{"add"})
		})
	})
}

func BenchmarkBuildDeviceBackendMatrix(b *testing.B) {
	for b.Loop() {
		matrix, err := BuildDeviceBackendMatrix()
		if err != nil {
			b.Fatal(err)
		}

		if len(matrix.Backends) != 3 {
			b.Fatalf("backends: got %d want 3", len(matrix.Backends))
		}
	}
}

func BenchmarkValidateDeviceBackendMatrix(b *testing.B) {
	matrix, err := BuildDeviceBackendMatrix()
	if err != nil {
		b.Fatal(err)
	}

	for b.Loop() {
		if validateErr := ValidateDeviceBackendMatrix(matrix); validateErr != nil {
			b.Fatal(validateErr)
		}
	}
}

func backendRow(matrix *DeviceBackendMatrix, name DeviceBackendName) DeviceBackendRow {
	for _, row := range matrix.Backends {
		if row.Backend == name {
			return row
		}
	}

	return DeviceBackendRow{}
}

func kernelRow(matrix *DeviceBackendMatrix, kernelName string) KernelCoverageRow {
	for _, row := range matrix.Kernels {
		if row.KernelName == kernelName {
			return row
		}
	}

	return KernelCoverageRow{}
}

func requiredOpMetal(matrix *DeviceBackendMatrix, operationID ir.OpType) KernelRegistrationStatus {
	for _, row := range matrix.RequiredOps {
		if row.OperationID == operationID {
			return row.MetalKernels
		}
	}

	return KernelNotRegistered
}
