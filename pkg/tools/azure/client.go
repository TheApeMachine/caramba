package azure

import (
	"os"

	"github.com/microsoft/azure-devops-go-api/azuredevops/v7"
	"github.com/theapemachine/caramba/pkg/datura"
)

/*
Client provides a high-level interface to Azure DevOps services.
It manages connections and operations for both work items and wiki pages
through a unified streaming interface.
*/
type Client struct {
	conn     *azuredevops.Connection
	workitem *WorkItem
	wiki     *Wiki
}

/*
NewClient creates a new Azure DevOps client using environment variables for authentication.

It initializes connections to work item and wiki services using a personal access token.
The client uses AZURE_ORG_URL and AZURE_PERSONAL_ACCESS_TOKEN environment variables.
*/
func NewClient() *Client {
	conn := azuredevops.NewPatConnection(
		os.Getenv("AZURE_ORG_URL"),
		os.Getenv("AZURE_PERSONAL_ACCESS_TOKEN"),
	)

	workitem := NewWorkItem(conn)
	wiki := NewWiki(conn)

	return &Client{
		conn:     conn,
		workitem: workitem,
		wiki:     wiki,
	}
}

func (c *Client) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)
	}()

	return out
}
