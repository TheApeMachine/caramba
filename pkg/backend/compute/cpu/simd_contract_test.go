package cpu

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

var avx2VectorPattern = regexp.MustCompile(
	`\bY[0-9]+\b|VMOVUPD|VADDPD|VMULPD|VFMADD|VBROADCAST|VPERM|VINSERT|VEXTRACT|VPADD|VPSUB|VPCM|VGATHER|VZEROUPPER`,
)
var sse2VectorPattern = regexp.MustCompile(
	`\bX[0-9]+\b|MOVUPD|ADDPD|MULPD|SUBPD|DIVPD|MAXPD|MINPD|SQRTPD|SHUFPD|PADD|PSUB|PCM|CVT`,
)
var neonVectorPattern = regexp.MustCompile(
	`\bV[0-9]+\.(D2|S4|B16|H8)|VLD1|VST1|VFADD|VFSUB|VFMUL|VFDIV|VFSQRT|VFMAX|VFMIN|VCMEQ|VCMGT|VAND|VORR|VEOR|VDUP`,
)

func TestSIMDAssemblyContract(test *testing.T) {
	Convey("Given the CPU SIMD assembly tree", test, func() {
		Convey("It should not advertise scalar aliases as SIMD implementations", func() {
			failures := findSIMDContractFailures("operation", "optimizer")

			So(failures, ShouldBeEmpty)
		})

		Convey("It should keep SSE2 assembly out of AVX2 files", func() {
			failures := findSIMDArchitectureSplitFailures("operation", "optimizer")

			So(failures, ShouldBeEmpty)
		})

		Convey("It should require ISA-named files to carry real vector kernels", func() {
			failures := findSIMDVectorKernelFailures("operation", "optimizer")

			So(failures, ShouldBeEmpty)
		})

		Convey("It should keep AVX2, SSE2, and NEON coverage aligned", func() {
			failures := findSIMDCoverageFailures("operation", "optimizer")

			So(failures, ShouldBeEmpty)
		})
	})
}

func findSIMDContractFailures(roots ...string) []string {
	var failures []string

	for _, root := range roots {
		_ = filepath.WalkDir(root, func(path string, dirEntry os.DirEntry, walkErr error) error {
			if walkErr != nil || dirEntry.IsDir() || filepath.Ext(path) != ".s" {
				return nil
			}

			body, err := os.ReadFile(path)
			if err != nil {
				failures = append(failures, path+": "+err.Error())
				return nil
			}

			text := string(body)

			for _, marker := range []string{
				"JMP ·scalar",
				"B ·scalar",
				"intentionally a stub",
				"alias to scalar",
			} {
				if strings.Contains(text, marker) {
					failures = append(failures, path+": contains "+marker)
				}
			}

			return nil
		})
	}

	return failures
}

func findSIMDVectorKernelFailures(roots ...string) []string {
	var failures []string

	for _, root := range roots {
		_ = filepath.WalkDir(root, func(path string, dirEntry os.DirEntry, walkErr error) error {
			if walkErr != nil || dirEntry.IsDir() || filepath.Ext(path) != ".s" {
				return nil
			}

			body, err := os.ReadFile(path)
			if err != nil {
				failures = append(failures, path+": "+err.Error())
				return nil
			}

			text := string(body)

			switch {
			case strings.HasSuffix(path, "_avx2_amd64.s"):
				failures = appendVectorKernelFailure(failures, path, text, avx2VectorPattern)
			case strings.HasSuffix(path, "_sse2_amd64.s"):
				failures = appendVectorKernelFailure(failures, path, text, sse2VectorPattern)

				if avx2VectorPattern.MatchString(text) {
					failures = append(failures, path+": contains AVX2 register or instruction")
				}
			case strings.HasSuffix(path, "_sse_amd64.s"):
				failures = append(failures, path+": SSE2 kernels must use _sse2_amd64.s")
			case strings.HasSuffix(path, "_neon_arm64.s"):
				failures = appendVectorKernelFailure(failures, path, text, neonVectorPattern)
			}

			return nil
		})
	}

	return failures
}

func findSIMDCoverageFailures(roots ...string) []string {
	coverage := make(map[string]map[string]bool)

	for _, root := range roots {
		_ = filepath.WalkDir(root, func(path string, dirEntry os.DirEntry, walkErr error) error {
			if walkErr != nil || dirEntry.IsDir() || filepath.Ext(path) != ".s" {
				return nil
			}

			base, isa := simdAssemblyBase(path)

			if base == "" {
				return nil
			}

			if coverage[base] == nil {
				coverage[base] = make(map[string]bool)
			}

			coverage[base][isa] = true

			return nil
		})
	}

	var failures []string

	for base, isaSet := range coverage {
		for _, isa := range []string{"avx2_amd64", "sse2_amd64", "neon_arm64"} {
			if !isaSet[isa] {
				failures = append(failures, base+": missing "+isa)
			}
		}
	}

	return failures
}

func simdAssemblyBase(path string) (string, string) {
	for _, isa := range []string{"avx2_amd64", "sse2_amd64", "neon_arm64"} {
		suffix := "_" + isa + ".s"

		if strings.HasSuffix(path, suffix) {
			return strings.TrimSuffix(path, suffix), isa
		}
	}

	return "", ""
}

func appendVectorKernelFailure(
	failures []string,
	path string,
	text string,
	vectorPattern *regexp.Regexp,
) []string {
	if !strings.Contains(text, "TEXT ") {
		return append(failures, path+": contains no TEXT symbol")
	}

	if !vectorPattern.MatchString(text) {
		return append(failures, path+": contains no vector register or vector instruction")
	}

	return failures
}

func findSIMDArchitectureSplitFailures(roots ...string) []string {
	var failures []string

	for _, root := range roots {
		_ = filepath.WalkDir(root, func(path string, dirEntry os.DirEntry, walkErr error) error {
			if walkErr != nil || dirEntry.IsDir() || filepath.Ext(path) != ".s" {
				return nil
			}

			body, err := os.ReadFile(path)
			if err != nil {
				failures = append(failures, path+": "+err.Error())
				return nil
			}

			text := string(body)

			if strings.HasSuffix(path, "_avx2_amd64.s") && strings.Contains(text, "SSE2(SB)") {
				failures = append(failures, path+": contains SSE2 TEXT symbol")
			}

			for _, marker := range []string{
				"defined in ",
				"implemented in ",
				"symbols live in the AVX2",
				"exists to satisfy",
				"intentionally empty",
			} {
				if strings.Contains(text, marker+"primitives_avx2_amd64.s") ||
					strings.Contains(text, marker+"the AVX2") ||
					strings.Contains(text, marker+"avx2_amd64.s") {
					failures = append(failures, path+": contains stale architecture split marker")
				}
			}

			return nil
		})
	}

	return failures
}
