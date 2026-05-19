package store_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStorePackagesDoNotCallOsGetenv(test *testing.T) {
	storeRoot := "."
	entries, readErr := os.ReadDir(storeRoot)

	if readErr != nil {
		test.Fatalf("read store root: %v", readErr)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		packageDir := filepath.Join(storeRoot, entry.Name())
		walkErr := filepath.WalkDir(packageDir, func(path string, dirEntry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}

			if dirEntry.IsDir() {
				return nil
			}

			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}

			contentBytes, readFileErr := os.ReadFile(path)

			if readFileErr != nil {
				return readFileErr
			}

			content := string(contentBytes)

			if strings.Contains(content, "os.Getenv") || strings.Contains(content, "os.LookupEnv") {
				test.Errorf("%s must not call os.Getenv or os.LookupEnv; use pkg/config", path)
			}

			return nil
		})

		if walkErr != nil {
			test.Fatalf("walk %s: %v", packageDir, walkErr)
		}
	}
}
