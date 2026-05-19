package coverageaudit

import (
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	. "github.com/smartystreets/goconvey/convey"
)

func TestBuildBackendCoverageMatrix(t *testing.T) {
	Convey("Given combined backend coverage audits", t, func() {
		matrix, err := BuildBackendCoverageMatrix()

		So(err, ShouldBeNil)
		So(matrix, ShouldNotBeNil)

		Convey("It should match T1.2 inventory counts", func() {
			So(matrix.Inventory.BackendMethods, ShouldEqual, 151)
			So(matrix.Inventory.RequiredOperations, ShouldEqual, 119)
			So(matrix.Inventory.DirectCrossLinks, ShouldEqual, 74)
			So(matrix.Inventory.CompositeCrossLinks, ShouldEqual, 3)
			So(matrix.Inventory.KernelRegistryCrossLinks, ShouldEqual, 32)
			So(matrix.Inventory.GraphOnlyCrossLinks, ShouldEqual, 10)
			So(matrix.Inventory.UnmappedBackendMethods, ShouldEqual, 88)
		})

		Convey("It should match T1.3 CPU dispatch counts", func() {
			So(matrix.CPU.Domains, ShouldEqual, 30)
			So(matrix.CPU.ScalarDomains, ShouldEqual, 30)
			So(matrix.CPU.AVX512Domains, ShouldEqual, 2)
			So(matrix.CPU.AVX2Domains, ShouldEqual, 2)
			So(matrix.CPU.SSE2Domains, ShouldEqual, 2)
			So(matrix.CPU.NEONDomains, ShouldEqual, 20)
			So(matrix.CPU.AMD64SIMDDomainNames, ShouldResemble, []string{"activation", "pospop"})
		})

		Convey("It should match T1.4 device backend counts", func() {
			So(matrix.Device.MetalKernelRegistrations, ShouldEqual, 462)
			So(matrix.Device.MetalUniqueKernelNames, ShouldEqual, 158)
			So(matrix.Device.MetalRequiredOpsRegistered, ShouldEqual, 68)
			So(matrix.Device.CUDAKernelRegistrations, ShouldEqual, 0)
			So(matrix.Device.XLAKernelRegistrations, ShouldEqual, 0)
		})

		Convey("It should list eight R1 execution targets", func() {
			So(len(matrix.Targets), ShouldEqual, 8)
			So(matrix.Targets[5].Target, ShouldEqual, "Metal")
			So(matrix.Targets[5].Registered, ShouldEqual, 68)
			So(matrix.Targets[5].Applicable, ShouldEqual, len(ir.RequiredOperationIDs()))
		})
	})
}

func TestValidateBackendCoverageMatrix(t *testing.T) {
	Convey("Given the combined coverage matrix", t, func() {
		matrix, err := BuildBackendCoverageMatrix()

		So(err, ShouldBeNil)

		Convey("It should validate structural invariants", func() {
			So(ValidateBackendCoverageMatrix(matrix), ShouldBeNil)
		})
	})
}

func TestRenderMarkdown(t *testing.T) {
	Convey("Given the combined coverage matrix", t, func() {
		matrix, err := BuildBackendCoverageMatrix()

		So(err, ShouldBeNil)

		Convey("It should render the T1.5 coverage document header", func() {
			doc := RenderMarkdown(matrix)

			So(doc, ShouldContainSubstring, "# Backend coverage matrix (T1.5)")
			So(doc, ShouldContainSubstring, "## R1 execution targets")
			So(doc, ShouldContainSubstring, "| Go scalar | CPU domains | 30 | 30 |")
			So(doc, ShouldContainSubstring, "| Metal | Required IR operations | 68 | 119 |")
			So(doc, ShouldContainSubstring, "[`backend-inventory.md`](./backend-inventory.md)")
		})
	})
}

func BenchmarkBuildBackendCoverageMatrix(b *testing.B) {
	for b.Loop() {
		matrix, err := BuildBackendCoverageMatrix()
		if err != nil {
			b.Fatal(err)
		}

		if matrix.Inventory.BackendMethods != 151 {
			b.Fatalf("backend methods: got %d want 151", matrix.Inventory.BackendMethods)
		}
	}
}

func BenchmarkValidateBackendCoverageMatrix(b *testing.B) {
	matrix, err := BuildBackendCoverageMatrix()
	if err != nil {
		b.Fatal(err)
	}

	for b.Loop() {
		if validateErr := ValidateBackendCoverageMatrix(matrix); validateErr != nil {
			b.Fatal(validateErr)
		}
	}
}
