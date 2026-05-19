package dispatchaudit

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestBuildCPUDispatchMatrix(t *testing.T) {
	Convey("Given pkg/backend/device/cpu operation domains", t, func() {
		matrix, err := BuildCPUDispatchMatrix()

		So(err, ShouldBeNil)
		So(matrix, ShouldNotBeNil)

		Convey("It should list 30 domains excluding cpu/neon", func() {
			So(len(matrix.Rows), ShouldEqual, 30)
		})

		Convey("It should register scalar on every domain", func() {
			for _, row := range matrix.Rows {
				So(row.Scalar, ShouldEqual, ISARegistered)
			}
		})

		Convey("It should match expected AVX-512 registration counts", func() {
			counts := summarize(matrix)

			So(counts[ISAPathAVX512], ShouldEqual, 12)
		})

		Convey("It should match expected AVX2 and SSE2 registration counts", func() {
			counts := summarize(matrix)

			So(counts[ISAPathAVX2], ShouldEqual, 2)
			So(counts[ISAPathSSE2], ShouldEqual, 2)
		})

		Convey("It should match expected NEON registration counts", func() {
			counts := summarize(matrix)

			So(counts[ISAPathNEON], ShouldEqual, 20)
		})

		Convey("It should register amd64 SIMD only on activation and pospop", func() {
			avx2Domains := domainNamesWith(matrix, ISAPathAVX2)

			So(avx2Domains, ShouldResemble, []string{"activation", "pospop"})
		})

		Convey("It should register AVX-512 only on activation and pospop", func() {
			avx512Domains := domainNamesWith(matrix, ISAPathAVX512)

			So(avx512Domains, ShouldResemble, []string{
				"activation", "attention", "convolution", "dot", "dropout", "elementwise", "embedding",
				"losses", "matmul", "pool", "pospop", "reduction",
			})
		})

		Convey("It should register NEON and AVX-512 on elementwise", func() {
			row := rowByDomain(matrix, "elementwise")

			So(row.NEON, ShouldEqual, ISARegistered)
			So(row.AVX512, ShouldEqual, ISARegistered)
			So(row.AVX2, ShouldEqual, ISANotRegistered)
			So(row.SSE2, ShouldEqual, ISANotRegistered)
		})

		Convey("It should register AVX-512 and NEON on dot", func() {
			row := rowByDomain(matrix, "dot")

			So(row.Scalar, ShouldEqual, ISARegistered)
			So(row.AVX512, ShouldEqual, ISARegistered)
			So(row.NEON, ShouldEqual, ISARegistered)
			So(row.AVX2, ShouldEqual, ISANotRegistered)
			So(row.SSE2, ShouldEqual, ISANotRegistered)
		})

		Convey("It should register AVX-512 and NEON on matmul", func() {
			row := rowByDomain(matrix, "matmul")

			So(row.Scalar, ShouldEqual, ISARegistered)
			So(row.AVX512, ShouldEqual, ISARegistered)
			So(row.NEON, ShouldEqual, ISARegistered)
			So(row.AVX2, ShouldEqual, ISANotRegistered)
			So(row.SSE2, ShouldEqual, ISANotRegistered)
		})

		Convey("It should register AVX-512 and NEON on reduction", func() {
			row := rowByDomain(matrix, "reduction")

			So(row.Scalar, ShouldEqual, ISARegistered)
			So(row.AVX512, ShouldEqual, ISARegistered)
			So(row.NEON, ShouldEqual, ISARegistered)
			So(row.AVX2, ShouldEqual, ISANotRegistered)
			So(row.SSE2, ShouldEqual, ISANotRegistered)
		})

		Convey("It should register AVX-512 and NEON on pool", func() {
			row := rowByDomain(matrix, "pool")

			So(row.Scalar, ShouldEqual, ISARegistered)
			So(row.AVX512, ShouldEqual, ISARegistered)
			So(row.NEON, ShouldEqual, ISARegistered)
			So(row.AVX2, ShouldEqual, ISANotRegistered)
			So(row.SSE2, ShouldEqual, ISANotRegistered)
		})

		Convey("It should register AVX-512 and NEON on dropout", func() {
			row := rowByDomain(matrix, "dropout")

			So(row.Scalar, ShouldEqual, ISARegistered)
			So(row.AVX512, ShouldEqual, ISARegistered)
			So(row.NEON, ShouldEqual, ISARegistered)
			So(row.AVX2, ShouldEqual, ISANotRegistered)
			So(row.SSE2, ShouldEqual, ISANotRegistered)
		})

		Convey("It should register AVX-512 and NEON on losses", func() {
			row := rowByDomain(matrix, "losses")

			So(row.Scalar, ShouldEqual, ISARegistered)
			So(row.AVX512, ShouldEqual, ISARegistered)
			So(row.NEON, ShouldEqual, ISARegistered)
			So(row.AVX2, ShouldEqual, ISANotRegistered)
			So(row.SSE2, ShouldEqual, ISANotRegistered)
		})

		Convey("It should register AVX-512 and NEON on convolution", func() {
			row := rowByDomain(matrix, "convolution")

			So(row.Scalar, ShouldEqual, ISARegistered)
			So(row.AVX512, ShouldEqual, ISARegistered)
			So(row.NEON, ShouldEqual, ISARegistered)
			So(row.AVX2, ShouldEqual, ISANotRegistered)
			So(row.SSE2, ShouldEqual, ISANotRegistered)
		})

		Convey("It should register AVX-512 and NEON on attention", func() {
			row := rowByDomain(matrix, "attention")

			So(row.Scalar, ShouldEqual, ISARegistered)
			So(row.AVX512, ShouldEqual, ISARegistered)
			So(row.NEON, ShouldEqual, ISARegistered)
			So(row.AVX2, ShouldEqual, ISANotRegistered)
			So(row.SSE2, ShouldEqual, ISANotRegistered)
		})

		Convey("It should register AVX-512 on embedding", func() {
			row := rowByDomain(matrix, "embedding")

			So(row.Scalar, ShouldEqual, ISARegistered)
			So(row.AVX512, ShouldEqual, ISARegistered)
			So(row.NEON, ShouldEqual, ISANotRegistered)
			So(row.AVX2, ShouldEqual, ISANotRegistered)
			So(row.SSE2, ShouldEqual, ISANotRegistered)
		})
	})
}

func TestValidateCPUDispatchMatrix(t *testing.T) {
	Convey("Given the CPU dispatch matrix", t, func() {
		matrix, err := BuildCPUDispatchMatrix()

		So(err, ShouldBeNil)

		Convey("It should validate structural invariants", func() {
			So(ValidateCPUDispatchMatrix(matrix), ShouldBeNil)
		})
	})
}

func TestRenderMarkdown(t *testing.T) {
	Convey("Given the CPU dispatch matrix", t, func() {
		matrix, err := BuildCPUDispatchMatrix()

		So(err, ShouldBeNil)

		Convey("It should render a markdown table header", func() {
			doc := RenderMarkdown(matrix)

			So(doc, ShouldContainSubstring, "# CPU dispatch matrix (T1.3)")
			So(doc, ShouldContainSubstring, "| Domain | Scalar | AVX-512 | AVX2 | SSE2 | NEON |")
			So(doc, ShouldContainSubstring, "| activation |")
		})
	})
}

func BenchmarkBuildCPUDispatchMatrix(b *testing.B) {
	for b.Loop() {
		matrix, err := BuildCPUDispatchMatrix()
		if err != nil {
			b.Fatal(err)
		}

		if len(matrix.Rows) != 30 {
			b.Fatalf("rows: got %d want 30", len(matrix.Rows))
		}
	}
}

func BenchmarkValidateCPUDispatchMatrix(b *testing.B) {
	matrix, err := BuildCPUDispatchMatrix()
	if err != nil {
		b.Fatal(err)
	}

	for b.Loop() {
		if validateErr := ValidateCPUDispatchMatrix(matrix); validateErr != nil {
			b.Fatal(validateErr)
		}
	}
}

func rowByDomain(matrix *CPUDispatchMatrix, domainName string) DomainDispatchRow {
	for _, row := range matrix.Rows {
		if row.Domain == domainName {
			return row
		}
	}

	return DomainDispatchRow{}
}
