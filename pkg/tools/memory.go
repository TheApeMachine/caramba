package tools

import (
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
)

func init() {
	provider.RegisterTool("memory")
}

/*
MemoryTool provides a unified interface for interacting with multiple memory stores.
*/
type MemoryTool struct {
	buffer *stream.Buffer
	stores map[string]io.ReadWriteCloser
	Schema *provider.Tool
}

func NewMemoryTool(stores map[string]io.ReadWriteCloser) *MemoryTool {
	errnie.Debug("NewMemoryTool")

	return &MemoryTool{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("MemoryTool.buffer")
			return nil
		}),
		stores: stores,
		Schema: provider.NewTool(
			provider.WithFunction(
				"memory",
				"A tool which can interact with various memory stores through a unified interface.",
			),
			provider.WithProperty(
				"question",
				"string",
				"A question which is used to retrieve information from a vector database.",
				[]any{},
			),
			provider.WithProperty(
				"keywords",
				"string",
				"A comma separated list of keywords which are used to retrieve information from all memory stores.",
				[]any{},
			),
			provider.WithProperty(
				"cypher",
				"string",
				"A Cypher query which is used to retrieve information from a graph database.",
				[]any{},
			),
		),
	}
}

func (mt *MemoryTool) Read(p []byte) (n int, err error) {
	errnie.Debug("MemoryTool.Read")
	return mt.buffer.Read(p)
}

func (mt *MemoryTool) Write(p []byte) (n int, err error) {
	errnie.Debug("MemoryTool.Write")
	return mt.buffer.Write(p)
}

func (mt *MemoryTool) Close() error {
	errnie.Debug("MemoryTool.Close")
	return mt.buffer.Close()
}
