//go:build amd64

package active_inference

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/complianceaudit"
)

func TestActiveInferenceAVX512AssemblyCompliance(t *testing.T) {
	convey.Convey("Given active_inference AVX-512 assembly sources", t, func() {
		matrix, err := complianceaudit.BuildComplianceMatrix()
		convey.So(err, convey.ShouldBeNil)

		var activeInferenceAVX512 []complianceaudit.Finding
		for _, finding := range matrix.Findings {
			if !strings.Contains(
				finding.Path,
				string(filepath.Separator)+"active_inference"+string(filepath.Separator),
			) {
				continue
			}

			if !strings.Contains(filepath.Base(finding.Path), "avx512") {
				continue
			}

			activeInferenceAVX512 = append(activeInferenceAVX512, finding)
		}

		convey.Convey("It should have zero compliance findings on avx512 amd64 kernels", func() {
			convey.So(len(activeInferenceAVX512), convey.ShouldEqual, 0)
		})
	})
}
