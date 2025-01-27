package provider

import (
	"context"
	"io"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/memory"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/errnie"
)

type Tool interface {
	Name() string
	Description() string
	GenerateSchema() interface{}
	Use(context.Context, map[string]any) string
	Connect(context.Context, io.ReadWriteCloser) error
}

func NewToolset(role string) []Tool {
	tools := []Tool{}

	for _, tool := range toolsets[role] {
		tools = append(tools, toolMap[tool])
	}

	if len(tools) == 0 {
		errnie.Warn("No tools found for role %s", role)
	}

	return tools
}

var toolsets = map[string][]string{
	"default": {
		"memory",
	},
	"manager": {
		"azure",
	},
	"researcher": {
		"browser",
	},
	"architect": {
		"azure",
		"github",
	},
	"developer": {
		"container",
		"github",
	},
	"tester": {
		"azure",
		"browser",
	},
	"reviewer": {
		"azure",
		"github",
	},
}

var qdrantCollection = viper.GetViper().GetString("tools.qdrant.collection")
var qdrantDimension = viper.GetViper().GetUint64("tools.qdrant.dimension")

var toolMap = map[string]Tool{
	"azure":   tools.NewAzure(),
	"browser": tools.NewBrowser(),
	// "container":    tools.NewContainer(),
	"github":       tools.NewGithub(),
	"neo4j_query":  tools.NewNeo4jQuery(),
	"neo4j_store":  tools.NewNeo4jStore(),
	"qdrant_query": tools.NewQdrantQuery(qdrantCollection, qdrantDimension),
	"qdrant_store": tools.NewQdrantStore(qdrantCollection, qdrantDimension),
	"memory":       memory.NewLongTerm(),
	"slack":        tools.NewSlack(),
	"trengo":       tools.NewTrengo(),
}
