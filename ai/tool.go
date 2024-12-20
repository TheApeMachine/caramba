package ai

import "io"

type Tool interface {
	GenerateSchema() string
	Use(map[string]any) string
	Connect(io.ReadWriteCloser)
}
