package tools

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools/trengo"
)

func init() {
	fmt.Println("tools.trengo.init")
	provider.RegisterTool("trengo")
}

type Trengo struct {
	client *trengo.Client
	Schema *provider.Tool
}

func NewTrengo() *Trengo {
	client := trengo.NewClient()

	return &Trengo{
		client: client,
		Schema: GetToolSchema("trengo"),
	}
}

func (t *Trengo) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)
	}()

	return out
}
