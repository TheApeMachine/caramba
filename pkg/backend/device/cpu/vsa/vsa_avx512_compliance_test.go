//go:build amd64

package vsa

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/complianceaudit"
)

func TestVSAAVX512AssemblyCompliance(t *testing.T) {
	convey.Convey("Given vsa AVX-512 assembly sources", t, func() {
		matrix, err := complianceaudit.BuildComplianceMatrix()
		convey.So(err, convey.ShouldBeNil)

		var vsaAVX512 []complianceaudit.Finding
		for _, finding := range matrix.Findings {
			if !strings.Contains(
				finding.Path,
				string(filepath.Separator)+"vsa"+string(filepath.Separator),
			) {
				continue
			}

			if !strings.Contains(filepath.Base(finding.Path), "avx512") {
				continue
			}

			vsaAVX512 = append(vsaAVX512, finding)
		}

		convey.Convey("It should have zero compliance findings on avx512 amd64 kernels", func() {
			convey.So(len(vsaAVX512), convey.ShouldEqual, 0)
		})
	})
}
