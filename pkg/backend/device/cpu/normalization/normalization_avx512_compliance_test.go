//go:build amd64

package normalization

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/complianceaudit"
)

func TestNormalizationAVX512AssemblyCompliance(t *testing.T) {
	convey.Convey("Given normalization AVX-512 assembly sources", t, func() {
		matrix, err := complianceaudit.BuildComplianceMatrix()
		convey.So(err, convey.ShouldBeNil)

		var normalizationAVX512 []complianceaudit.Finding
		for _, finding := range matrix.Findings {
			if !strings.Contains(finding.Path, string(filepath.Separator)+"normalization"+string(filepath.Separator)) {
				continue
			}

			if !strings.Contains(filepath.Base(finding.Path), "avx512") {
				continue
			}

			normalizationAVX512 = append(normalizationAVX512, finding)
		}

		convey.Convey("It should have zero compliance findings on avx512 amd64 kernels", func() {
			convey.So(len(normalizationAVX512), convey.ShouldEqual, 0)
		})
	})
}
