package complianceaudit

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestBuildComplianceMatrix(t *testing.T) {
	Convey("Given pkg/backend/device/cpu", t, func() {
		matrix, err := BuildComplianceMatrix()

		So(err, ShouldBeNil)
		So(matrix, ShouldNotBeNil)

		Convey("It should report zero compliance findings", func() {
			So(len(matrix.Findings), ShouldEqual, 0)
		})
	})
}

func TestValidateComplianceMatrix(t *testing.T) {
	Convey("Given the compliance matrix", t, func() {
		matrix, err := BuildComplianceMatrix()

		So(err, ShouldBeNil)

		Convey("It should validate with no findings", func() {
			So(ValidateComplianceMatrix(matrix), ShouldBeNil)
		})
	})
}

func TestRenderMarkdown(t *testing.T) {
	Convey("Given a clean compliance matrix", t, func() {
		matrix := &ComplianceMatrix{}

		Convey("It should render the no-findings header", func() {
			doc := RenderMarkdown(matrix)

			So(doc, ShouldContainSubstring, "# Backend compliance audit (T1.6)")
			So(doc, ShouldContainSubstring, "No findings.")
		})
	})
}

func BenchmarkBuildComplianceMatrix(b *testing.B) {
	for b.Loop() {
		matrix, err := BuildComplianceMatrix()
		if err != nil {
			b.Fatal(err)
		}

		if len(matrix.Findings) != 0 {
			b.Fatalf("findings: got %d want 0", len(matrix.Findings))
		}
	}
}

func BenchmarkValidateComplianceMatrix(b *testing.B) {
	matrix, err := BuildComplianceMatrix()
	if err != nil {
		b.Fatal(err)
	}

	for b.Loop() {
		if validateErr := ValidateComplianceMatrix(matrix); validateErr != nil {
			b.Fatal(validateErr)
		}
	}
}
