package complianceaudit

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

/*
ComplianceMatrix is the full audit snapshot for pkg/backend/device/cpu.
*/
type ComplianceMatrix struct {
	Findings []Finding
}

/*
BuildComplianceMatrix scans pkg/backend/device/cpu.
*/
func BuildComplianceMatrix() (*ComplianceMatrix, error) {
	cpuRoot, err := locateCPURoot()
	if err != nil {
		return nil, err
	}

	findings, scanErr := ScanCPUTree(cpuRoot)
	if scanErr != nil {
		return nil, scanErr
	}

	return &ComplianceMatrix{Findings: findings}, nil
}

/*
ValidateComplianceMatrix returns an error when any finding remains.
*/
func ValidateComplianceMatrix(matrix *ComplianceMatrix) error {
	if matrix == nil {
		return fmt.Errorf("complianceaudit: nil matrix")
	}

	if len(matrix.Findings) > 0 {
		first := matrix.Findings[0]

		return fmt.Errorf(
			"complianceaudit: %d findings (first %s:%d %s: %s)",
			len(matrix.Findings),
			first.Path,
			first.Line,
			first.Kind,
			first.Summary,
		)
	}

	return nil
}

/*
RenderMarkdown emits a human-readable compliance report.
*/
func RenderMarkdown(matrix *ComplianceMatrix) string {
	if matrix == nil {
		return ""
	}

	var builder strings.Builder

	builder.WriteString("# Backend compliance audit (T1.6)\n\n")
	builder.WriteString("Machine-checkable source: `pkg/backend/device/complianceaudit/`.\n\n")

	if len(matrix.Findings) == 0 {
		builder.WriteString("No findings.\n")

		return builder.String()
	}

	builder.WriteString("| Kind | File | Line | Summary |\n")
	builder.WriteString("|------|------|-----:|---------|\n")

	for _, finding := range matrix.Findings {
		builder.WriteString(fmt.Sprintf(
			"| %s | `%s` | %d | %s |\n",
			finding.Kind,
			finding.Path,
			finding.Line,
			finding.Summary,
		))
	}

	return builder.String()
}

func locateCPURoot() (string, error) {
	_, filePath, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("complianceaudit: runtime.Caller failed")
	}

	root := filepath.Clean(filepath.Join(filepath.Dir(filePath), "..", "cpu"))

	info, statErr := os.Stat(root)
	if statErr != nil {
		return "", statErr
	}

	if !info.IsDir() {
		return "", fmt.Errorf("complianceaudit: cpu root is not a directory: %s", root)
	}

	return root, nil
}
