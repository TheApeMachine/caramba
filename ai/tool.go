package ai

import "io"

type Tool interface {
	GenerateSchema() string
	Initialize()
	Use(map[string]any) string
	Connect(io.ReadWriteCloser)
}
