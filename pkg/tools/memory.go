package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/memory"
	"github.com/theapemachine/caramba/pkg/provider"
)

func init() {
	fmt.Println("tools.memory.init")
	provider.RegisterTool("memory")
}

/*
MemoryTool provides a unified interface for interacting with multiple memory stores.
It handles both vector-based (Qdrant) and graph-based (Neo4j) storage systems.
*/
type MemoryTool struct {
	ctx    context.Context
	cancel context.CancelFunc
	stores map[string]interface{} // Map of store types to their implementations
	Schema *provider.Tool
}

// NewMemoryTool creates a new memory tool with the specified stores.
// If no stores are provided, it initializes with default Qdrant and Neo4j stores.
func NewMemoryTool() *MemoryTool {
	errnie.Debug("NewMemoryTool")

	ctx, cancel := context.WithCancel(context.Background())

	// Initialize stores
	storeMap := make(map[string]interface{})
	storeMap["vector"] = memory.NewQdrant()
	storeMap["graph"] = memory.NewNeo4j()

	return &MemoryTool{
		ctx:    ctx,
		cancel: cancel,
		stores: storeMap,
		Schema: GetToolSchema("memory"),
	}
}

// Generate implements the Generator pattern for MemoryTool.
// It processes queries and returns results through the artifact channel.
func (mt *MemoryTool) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	errnie.Debug("tools.MemoryTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-mt.ctx.Done():
			errnie.Debug("tools.MemoryTool.Generate.ctx.Done")
			mt.cancel()
			return
		case artifact := <-buffer:
			var results []map[string]any

			// Process each store with its relevant metadata
			for storeType, store := range mt.stores {
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

				// Check if we have relevant metadata for this store type
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
					var storeOutput chan *datura.Artifact

					// Use the appropriate Generate method based on store type
					switch storeType {
					case "vector":
						qdrantStore := store.(*memory.Qdrant)
						inputChan := make(chan *datura.Artifact, 1)
						inputChan <- storeArtifact
						storeOutput = qdrantStore.Generate(inputChan)
					case "graph":
						neo4jStore := store.(*memory.Neo4j)
						inputChan := make(chan *datura.Artifact, 1)
						inputChan <- storeArtifact
						storeOutput = neo4jStore.Generate(inputChan)
					}

					// Process store output
					if storeOutput != nil {
						for resultArtifact := range storeOutput {
							// Check for errors in the payload
							payload, err := resultArtifact.DecryptPayload()
							if err != nil {
								errnie.Error(err)
								continue
							}

							// Skip processing if we have an empty payload
							if len(payload) == 0 {
								continue
							}

							output := datura.GetMetaValue[string](resultArtifact, "output")
							if output != "" {
								var storeResults []map[string]any
								if err := json.Unmarshal([]byte(output), &storeResults); err == nil {
									results = append(results, storeResults...)
								}
							}
						}
					}
				}
			}

			// Set combined results in artifact metadata
			artifact.SetMetaValue("output", results)
			out <- artifact
		}
	}()

	return out
}
