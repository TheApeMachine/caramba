//go:build amd64

package layernorm

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/complianceaudit"
)

func TestLayerNormAVX512AssemblyCompliance(t *testing.T) {
	convey.Convey("Given layernorm AVX-512 assembly sources", t, func() {
		matrix, err := complianceaudit.BuildComplianceMatrix()
		convey.So(err, convey.ShouldBeNil)

		var layernormAVX512 []complianceaudit.Finding
		for _, finding := range matrix.Findings {
			if !strings.Contains(finding.Path, string(filepath.Separator)+"layernorm"+string(filepath.Separator)) {
				continue
			}

			if !strings.Contains(filepath.Base(finding.Path), "avx512") {
				continue
			}

			layernormAVX512 = append(layernormAVX512, finding)
		}

		convey.Convey("It should have zero compliance findings on avx512 amd64 kernels", func() {
			convey.So(len(layernormAVX512), convey.ShouldEqual, 0)
		})
	})
}
