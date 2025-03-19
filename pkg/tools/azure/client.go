package azure

import (
	"os"

	"github.com/microsoft/azure-devops-go-api/azuredevops/v7"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Client struct {
	buffer   *stream.Buffer
	conn     *azuredevops.Connection
	workitem *WorkItem
	wiki     *Wiki
}

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

			switch artifact.Role() {
			case uint32(datura.ArtifactRoleCreateWorkItem):
				return workitem.CreateWorkItem(artifact)
			case uint32(datura.ArtifactRoleUpdateWorkItem):
				return workitem.UpdateWorkItem(artifact)
			case uint32(datura.ArtifactRoleGetWorkItem):
				return workitem.GetWorkItem(artifact)
			case uint32(datura.ArtifactRoleListWorkItems):
				return workitem.ListWorkItems(artifact)
			case uint32(datura.ArtifactRoleCreateWikiPage):
				return wiki.CreatePage(artifact)
			case uint32(datura.ArtifactRoleUpdateWikiPage):
				return wiki.UpdatePage(artifact)
			case uint32(datura.ArtifactRoleGetWikiPage):
				return wiki.GetPage(artifact)
			case uint32(datura.ArtifactRoleListWikiPages):
				return wiki.ListPages(artifact)
			}

			return nil
		}),
		conn:     conn,
		workitem: workitem,
		wiki:     wiki,
	}
}

func (client *Client) Read(p []byte) (n int, err error) {
	errnie.Debug("azure.Client.Read")
	return client.buffer.Read(p)
}

func (client *Client) Write(p []byte) (n int, err error) {
	errnie.Debug("azure.Client.Write")
	return client.buffer.Write(p)
}

func (client *Client) Close() error {
	errnie.Debug("azure.Client.Close")
	return client.buffer.Close()
}
