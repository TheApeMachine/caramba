package tools

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools/github"
)

/*
init registers the GitHub tool with the provider system.
*/
func init() {
	fmt.Println("tools.github.init")
	provider.RegisterTool("github")
}

/*
Github provides a streaming interface to GitHub operations.
It manages GitHub API interactions through a buffered client connection
and implements io.ReadWriteCloser for streaming data processing.
*/
type Github struct {
	client *github.Client
	Schema *provider.Tool
}

/*
NewGithub creates a new GitHub tool instance.

It initializes a GitHub client and sets up a buffered stream for
processing GitHub operations. The buffer copies data bidirectionally
between the artifact and the GitHub client.
*/
func NewGithub() *Github {
	client := github.NewClient()

	return &Github{
		client: client,
		Schema: GetToolSchema("github"),
	}
}

func (g *Github) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)
	}()

	return out
}
