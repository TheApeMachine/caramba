package backendaudit

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteDeviceBackendMatrixDoc(t *testing.T) {
	if os.Getenv("CARAMBA_WRITE_DEVICE_BACKEND_DOC") == "" {
		t.Skip("set CARAMBA_WRITE_DEVICE_BACKEND_DOC=1 to regenerate docs/device-backend-matrix.md")
	}

	matrix, err := BuildDeviceBackendMatrix()
	if err != nil {
		t.Fatal(err)
	}

	deviceRoot, rootErr := locateDeviceBackendRoot()
	if rootErr != nil {
		t.Fatal(rootErr)
	}

	docPath := filepath.Join(deviceRoot, "..", "..", "..", "docs", "device-backend-matrix.md")
	doc := RenderMarkdown(matrix)

	writeErr := os.WriteFile(docPath, []byte(doc), 0o644)
	if writeErr != nil {
		t.Fatal(writeErr)
	}
}
