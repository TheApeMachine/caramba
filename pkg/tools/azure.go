package tools

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools/github"
)

func init() {
	fmt.Println("tools.azure.init")
	provider.RegisterTool("azure")
}

/*
Azure represents the Azure DevOps integration tool that provides functionality
for interacting with Azure DevOps services. It implements io.ReadWriteCloser
interface for streaming data through the tool.
*/
type Azure struct {
	client *github.Client
	Schema *provider.Tool
}

/*
NewAzure creates and initializes a new Azure tool instance with a configured
GitHub client and buffer for handling data streams.

Returns a new Azure tool instance ready for use.
*/
func NewAzure() *Azure {
	client := github.NewClient()

	return &Azure{
		client: client,
		Schema: GetToolSchema("azure"),
	}
}

func (a *Azure) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)
	}()

	return out
}
