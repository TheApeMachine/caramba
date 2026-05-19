//go:build amd64

package embedding

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/complianceaudit"
)

func TestEmbeddingAVX512AssemblyCompliance(t *testing.T) {
	convey.Convey("Given embedding AVX-512 assembly sources", t, func() {
		matrix, err := complianceaudit.BuildComplianceMatrix()
		convey.So(err, convey.ShouldBeNil)

		var embeddingAVX512 []complianceaudit.Finding
		for _, finding := range matrix.Findings {
			if !strings.Contains(finding.Path, string(filepath.Separator)+"embedding"+string(filepath.Separator)) {
				continue
			}

			if !strings.Contains(filepath.Base(finding.Path), "avx512") {
				continue
			}

			embeddingAVX512 = append(embeddingAVX512, finding)
		}

		convey.Convey("It should have zero compliance findings on avx512 amd64 kernels", func() {
			convey.So(len(embeddingAVX512), convey.ShouldEqual, 0)
		})
	})
}
