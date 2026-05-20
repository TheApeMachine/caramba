//go:build amd64

package predictive_coding

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/complianceaudit"
)

func TestPredictiveCodingAVX512AssemblyCompliance(t *testing.T) {
	convey.Convey("Given predictive_coding AVX-512 assembly sources", t, func() {
		matrix, err := complianceaudit.BuildComplianceMatrix()
		convey.So(err, convey.ShouldBeNil)

		var predictiveCodingAVX512 []complianceaudit.Finding
		for _, finding := range matrix.Findings {
			if !strings.Contains(
				finding.Path,
				string(filepath.Separator)+"predictive_coding"+string(filepath.Separator),
			) {
				continue
			}

			if !strings.Contains(filepath.Base(finding.Path), "avx512") {
				continue
			}

			predictiveCodingAVX512 = append(predictiveCodingAVX512, finding)
		}

		convey.Convey("It should have zero compliance findings on avx512 amd64 kernels", func() {
			convey.So(len(predictiveCodingAVX512), convey.ShouldEqual, 0)
		})
	})
}
