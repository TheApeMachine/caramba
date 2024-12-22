package ai

import "io"

type Tool interface {
	GenerateSchema() string
	Initialize() error
	Use(map[string]any) string
	Connect(io.ReadWriteCloser) error
}
