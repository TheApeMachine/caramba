package tools

import (
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/memory"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
)

func init() {
	provider.RegisterTool("memory")
}

/*
MemoryTool provides a unified interface for interacting with multiple memory stores.
It handles both vector-based (Qdrant) and graph-based (Neo4j) storage systems.
*/
type MemoryTool struct {
	buffer *stream.Buffer
	stores map[string]io.ReadWriteCloser
	Schema *provider.Tool
}

func NewMemoryTool(stores ...io.ReadWriteCloser) *MemoryTool {
	errnie.Debug("NewMemoryTool")

	// Initialize default stores if none provided
	if len(stores) == 0 {
		stores = []io.ReadWriteCloser{
			memory.NewQdrant(),
			memory.NewNeo4j(),
		}
	}

	// Map stores by type for easier access
	storeMap := make(map[string]io.ReadWriteCloser)
	for _, store := range stores {
		switch store.(type) {
		case *memory.Qdrant:
			storeMap["vector"] = store
		case *memory.Neo4j:
			storeMap["graph"] = store
		}
	}

	return &MemoryTool{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("MemoryTool.buffer")

			var results []map[string]any

			// Process each store with its relevant metadata
			for storeType, store := range storeMap {
				// Create store-specific artifact with relevant metadata
				storeArtifact := datura.New()
				switch storeType {
				case "vector":
					if q := datura.GetMetaValue[string](artifact, "question"); q != "" {
						storeArtifact.SetMetaValue("question", q)
					}
					if d := datura.GetMetaValue[string](artifact, "documents"); d != "" {
						storeArtifact.SetMetaValue("documents", d)
					}
				case "graph":
					if k := datura.GetMetaValue[string](artifact, "keywords"); k != "" {
						storeArtifact.SetMetaValue("keywords", k)
					}
					if c := datura.GetMetaValue[string](artifact, "cypher"); c != "" {
						storeArtifact.SetMetaValue("cypher", c)
					}
				}

				// Only process if the store artifact has any metadata set
				hasMetadata := false
				switch storeType {
				case "vector":
					q := datura.GetMetaValue[string](storeArtifact, "question")
					d := datura.GetMetaValue[string](storeArtifact, "documents")
					hasMetadata = q != "" || d != ""
				case "graph":
					k := datura.GetMetaValue[string](storeArtifact, "keywords")
					c := datura.GetMetaValue[string](storeArtifact, "cypher")
					hasMetadata = k != "" || c != ""
				}

				if hasMetadata {
					if _, err = io.Copy(store, storeArtifact); err != nil {
						return errnie.Error(err)
					}

					// Read results
					resultBytes := make([]byte, 1024*1024)
					if _, err = store.Read(resultBytes); err != nil && err != io.EOF {
						return errnie.Error(err)
					}

					var storeResults []map[string]any
					if err = json.Unmarshal(resultBytes, &storeResults); err == nil {
						results = append(results, storeResults...)
					}
				}
			}

			// Set combined results in artifact metadata
			artifact.SetMetaValue("output", results)
			return nil
		}),
		stores: storeMap,
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
				"A Cypher query which is used to retrieve from or store to a graph database. Useful when you're dealing with relational data.",
				[]any{},
			),
			provider.WithProperty(
				"documents",
				"string",
				`A JSON array of documents to store in the vector database. Follow this format: [{"content": "...", "metadata": {"<key>": "<value>", ...}, ...}]`,
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
	// Close all stores
	for _, store := range mt.stores {
		if err := store.Close(); err != nil {
			errnie.Error(err)
		}
	}
	return mt.buffer.Close()
}
