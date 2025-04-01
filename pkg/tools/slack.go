package tools

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools/slack"
)

/*
init registers the Slack tool with the provider system.
*/
func init() {
	fmt.Println("tools.slack.init")
	provider.RegisterTool("slack")
}

/*
Slack provides a streaming interface to Slack operations.
It manages Slack API interactions through a buffered client connection
and implements io.ReadWriteCloser for streaming data processing.
*/
type Slack struct {
	Schema *provider.Tool
	client *slack.Client
}

/*
NewSlack creates a new Slack tool instance.

It initializes a Slack client and sets up a buffered stream for
processing Slack operations. The buffer copies data bidirectionally
between the artifact and the Slack client.
*/
func NewSlack() *Slack {
	client := slack.NewClient()

	return &Slack{
		client: client,
		Schema: GetToolSchema("slack"),
	}
}

func (s *Slack) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	return s.client.Generate(buffer)
}
