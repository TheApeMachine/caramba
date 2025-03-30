package azure

import (
	"os"

	"github.com/microsoft/azure-devops-go-api/azuredevops/v7"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
Client provides a high-level interface to Azure DevOps services.
It manages connections and operations for both work items and wiki pages
through a unified streaming interface.
*/
type Client struct {
	buffer   *stream.Buffer
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
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("azure.Client.buffer")

			operation := datura.GetMetaValue[string](artifact, "operation")

			switch operation {
			case "create_work_item":
				return workitem.CreateWorkItem(artifact)
			case "update_work_item":
				return workitem.UpdateWorkItem(artifact)
			case "get_work_item":
				return workitem.GetWorkItem(artifact)
			case "list_work_items":
				return workitem.ListWorkItems(artifact)
			case "create_wiki_page":
				return wiki.CreatePage(artifact)
			case "update_wiki_page":
				return wiki.UpdatePage(artifact)
			case "get_wiki_page":
				return wiki.GetPage(artifact)
			case "list_wiki_pages":
				return wiki.ListPages(artifact)
			}

			return nil
		}),
		conn:     conn,
		workitem: workitem,
		wiki:     wiki,
	}
}

/*
Read implements the io.Reader interface.

It reads processed data from the internal buffer after Azure DevOps operations
have been completed.
*/
func (client *Client) Read(p []byte) (n int, err error) {
	errnie.Debug("azure.Client.Read")
	return client.buffer.Read(p)
}

/*
Write implements the io.Writer interface.

It writes operation requests to the internal buffer for processing by
the appropriate Azure DevOps service (work items or wiki).
*/
func (client *Client) Write(p []byte) (n int, err error) {
	errnie.Debug("azure.Client.Write")
	return client.buffer.Write(p)
}

/*
Close implements the io.Closer interface.

It cleans up resources by closing the internal buffer.
*/
func (client *Client) Close() error {
	errnie.Debug("azure.Client.Close")
	return client.buffer.Close()
}
