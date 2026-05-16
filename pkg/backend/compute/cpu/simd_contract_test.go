package cpu

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
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
