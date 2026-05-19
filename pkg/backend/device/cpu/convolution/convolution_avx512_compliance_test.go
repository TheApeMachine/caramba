//go:build amd64

package convolution

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/complianceaudit"
)

func TestConvolutionAVX512AssemblyCompliance(t *testing.T) {
	convey.Convey("Given convolution AVX-512 assembly sources", t, func() {
		matrix, err := complianceaudit.BuildComplianceMatrix()
		convey.So(err, convey.ShouldBeNil)

		var convolutionAVX512 []complianceaudit.Finding
		for _, finding := range matrix.Findings {
			if !strings.Contains(finding.Path, string(filepath.Separator)+"convolution"+string(filepath.Separator)) {
				continue
			}

			if !strings.Contains(filepath.Base(finding.Path), "avx512") {
				continue
			}

			convolutionAVX512 = append(convolutionAVX512, finding)
		}

		convey.Convey("It should have zero compliance findings on avx512 amd64 kernels", func() {
			convey.So(len(convolutionAVX512), convey.ShouldEqual, 0)
		})
	})
}
