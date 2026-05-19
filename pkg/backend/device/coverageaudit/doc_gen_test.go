package coverageaudit

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestWriteBackendCoverageDoc(t *testing.T) {
	if os.Getenv("CARAMBA_WRITE_BACKEND_COVERAGE_DOC") == "" {
		t.Skip("set CARAMBA_WRITE_BACKEND_COVERAGE_DOC=1 to regenerate docs/backend-coverage.md")
	}

	matrix, err := BuildBackendCoverageMatrix()
	if err != nil {
		t.Fatal(err)
	}

	deviceRoot, rootErr := locateDevicePackageRoot()
	if rootErr != nil {
		t.Fatal(rootErr)
	}

	docPath := filepath.Join(deviceRoot, "..", "..", "..", "docs", "backend-coverage.md")
	doc := RenderMarkdown(matrix)

	writeErr := os.WriteFile(docPath, []byte(doc), 0o644)
	if writeErr != nil {
		t.Fatal(writeErr)
	}
}

func locateDevicePackageRoot() (string, error) {
	_, filePath, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("coverageaudit: runtime.Caller failed")
	}

	root := filepath.Clean(filepath.Join(filepath.Dir(filePath), ".."))

	info, statErr := os.Stat(root)
	if statErr != nil {
		return "", statErr
	}

	if !info.IsDir() {
		return "", fmt.Errorf("coverageaudit: device root is not a directory: %s", root)
	}

	return root, nil
}
