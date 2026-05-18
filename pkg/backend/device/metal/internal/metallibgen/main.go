package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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
	if err := generator.Run("xcrun", generator.MetalArgs()...); err != nil {
		return err
	}

	return generator.Run("xcrun", generator.MetallibArgs()...)
}

func (generator *Generator) MetalArgs() []string {
	return []string{
		"-sdk",
		"macosx",
		"metal",
		"-c",
		filepath.Join(generator.packageDir, "add_float32.metal"),
		"-o",
		filepath.Join(generator.tempDir, "add_float32.air"),
	}
}

func (generator *Generator) MetallibArgs() []string {
	return []string{
		"-sdk",
		"macosx",
		"metallib",
		filepath.Join(generator.tempDir, "add_float32.air"),
		"-o",
		filepath.Join(generator.packageDir, "kernels.metallib"),
	}
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
