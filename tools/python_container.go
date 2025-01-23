package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

type PythonContainer struct {
	container *Container
}

func NewPythonContainer() *PythonContainer {
	return &PythonContainer{
		container: NewContainer(),
	}
}

func (p *PythonContainer) Name() string {
	return "python_container"
}

func (p *PythonContainer) Description() string {
	return "Execute Python code in an isolated container"
}

func (p *PythonContainer) Initialize() error {
	return p.container.Initialize()
}

func (p *PythonContainer) ExecutePython(ctx context.Context, code string) (string, error) {
	// Create temporary Python file
	tmpDir := "/tmp/python_exec"
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		return "", err
	}

	tmpFile := filepath.Join(tmpDir, "script.py")
	if err := os.WriteFile(tmpFile, []byte(code), 0644); err != nil {
		return "", err
	}

	// Execute Python script in container
	result := p.container.Use(ctx, map[string]any{
		"command": fmt.Sprintf("python3 %s", tmpFile),
	})

	return result, nil
}
