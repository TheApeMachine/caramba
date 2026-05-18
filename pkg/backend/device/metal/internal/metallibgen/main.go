package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

func main() {
	packageDir, err := os.Getwd()
	if err != nil {
		fatal(err)
	}

	tempDir, err := os.MkdirTemp("", "caramba-metal-*")
	if err != nil {
		fatal(err)
	}
	defer func() {
		_ = os.RemoveAll(tempDir)
	}()

	generator := NewGenerator(packageDir, tempDir)

	if err := generator.Generate(); err != nil {
		fatal(err)
	}
}

type Generator struct {
	packageDir string
	tempDir    string
}

func NewGenerator(packageDir string, tempDir string) *Generator {
	return &Generator{
		packageDir: packageDir,
		tempDir:    tempDir,
	}
}

func (generator *Generator) Generate() error {
	sources, err := generator.SourceFiles()
	if err != nil {
		return err
	}

	for _, source := range sources {
		if err := generator.Run("xcrun", generator.MetalArgs(source)...); err != nil {
			return err
		}
	}

	return generator.Run("xcrun", generator.MetallibArgs(sources)...)
}

func (generator *Generator) SourceFiles() ([]string, error) {
	pattern := filepath.Join(generator.packageDir, "*.metal")
	sources, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}

	sort.Strings(sources)

	if len(sources) == 0 {
		return nil, fmt.Errorf("no Metal source files matched %s", pattern)
	}

	return sources, nil
}

func (generator *Generator) MetalArgs(source string) []string {
	return []string{
		"-sdk",
		"macosx",
		"metal",
		"-c",
		source,
		"-o",
		generator.AirPath(source),
	}
}

func (generator *Generator) MetallibArgs(sources []string) []string {
	args := []string{
		"-sdk",
		"macosx",
		"metallib",
	}

	for _, source := range sources {
		args = append(args, generator.AirPath(source))
	}

	return append(
		args,
		"-o",
		filepath.Join(generator.packageDir, "kernels.metallib"),
	)
}

func (generator *Generator) AirPath(source string) string {
	base := filepath.Base(source)
	name := strings.TrimSuffix(base, filepath.Ext(base))

	return filepath.Join(generator.tempDir, name+".air")
}

func (generator *Generator) Run(name string, args ...string) error {
	command := exec.Command(name, args...)
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr

	if err := command.Run(); err != nil {
		return fmt.Errorf("%s %v: %w", name, args, err)
	}

	return nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
