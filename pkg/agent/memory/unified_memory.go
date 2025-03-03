/*
Package memory provides various memory storage implementations for Caramba agents.
This package offers different memory storage solutions, from simple in-memory
stores to sophisticated vector and graph-based memory implementations that
support semantic search and complex relationships between memory items.
*/
package memory

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/caramba/pkg/agent/tools"
	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
	"github.com/theapemachine/caramba/pkg/process"
)

type UnifiedMemory struct {
	logger      *output.Logger
	hub         *hub.Queue
	stores      map[string]Store
	embedder    EmbeddingProvider
	queryAgent  core.Agent
	mutateAgent core.Agent
}

func NewUnifiedMemory() *UnifiedMemory {
	buildAgent := func(name string, proc process.StructuredOutput) core.Agent {
		return core.NewAgentBuilder(
			name + "_agent",
		).WithLLM(
			llm.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "gpt-4o-mini"),
		).WithSystemPrompt(
			viper.GetViper().GetString("templates.memory_manager." + name),
		).WithProcess(
			proc,
		).WithIterationLimit(
			3,
		).WithTools(
			tools.NewSystem(),
		).WithStreaming(
			true,
		).Build()
	}

	// Create the embedding provider
	embedder := NewOpenAIEmbeddingProvider(
		os.Getenv("OPENAI_API_KEY"), "text-embedding-3-large",
	)

	return &UnifiedMemory{
		logger: output.NewLogger(),
		hub:    hub.NewQueue(),
		stores: map[string]Store{
			"vector": NewQDrantStore("long-term-memory", embedder),
			"graph":  NewNeo4jStore("neo4j"),
		},
		embedder:    embedder,
		queryAgent:  buildAgent("query", &process.MemoryLookup{}),
		mutateAgent: buildAgent("mutate", &process.MemoryMutate{}),
	}
}

func (memory *UnifiedMemory) SetParent(parent core.Agent) {
	memory.queryAgent.SetParent(parent)
	memory.mutateAgent.SetParent(parent)
}

func (memory *UnifiedMemory) QueryAgent() core.Agent {
	return memory.queryAgent
}

func (memory *UnifiedMemory) MutateAgent() core.Agent {
	return memory.mutateAgent
}

/*
Query uses the LLM to generate lookup queries for each store. It then executes
the queries and construct the string to inject into the agent's context.
*/
func (memory *UnifiedMemory) Query(ctx context.Context, proc *process.MemoryLookup) (string, error) {
	// Generate queries for each store
	var results strings.Builder

	for _, question := range proc.Questions {
		result, err := memory.stores["vector"].Query(ctx, map[string]any{
			"query": question,
			"limit": 3,
		})

		if err != nil {
			memory.logger.Error("unified_memory", err)
			return "", err
		}

		results.WriteString(fmt.Sprintf("%s: %s\n", question, result))
	}

	for _, keyword := range proc.Keywords {
		result, err := memory.stores["graph"].Query(ctx, map[string]any{
			"keywords": keyword,
		})

		if err != nil {
			memory.logger.Error("unified_memory", err)
			return "", err
		}

		results.WriteString(fmt.Sprintf("%s: %s\n", keyword, result))
	}

	if proc.Cypher != "" {
		result, err := memory.stores["graph"].Query(ctx, map[string]any{
			"query": proc.Cypher,
		})

		if err != nil {
			memory.logger.Error("unified_memory", err)
			return "", err
		}

		results.WriteString(fmt.Sprintf("%s: %s\n", proc.Cypher, result))
	}

	return results.String(), nil
}

func (memory *UnifiedMemory) Mutate(ctx context.Context, proc *process.MemoryMutate) error {
	for _, doc := range proc.Documents {
		memory.stores["vector"].Mutate(ctx, map[string]any{
			"content": doc,
		})
	}

	for _, relation := range proc.Relationships {
		memory.stores["graph"].Mutate(ctx, map[string]any{
			"relationship": relation,
		})
	}

	return nil
}
