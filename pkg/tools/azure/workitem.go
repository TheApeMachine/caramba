package azure

import (
	"bytes"
	"context"
	"encoding/json"

	"github.com/microsoft/azure-devops-go-api/azuredevops/v7"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7/webapi"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7/workitemtracking"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
WorkItem manages Azure DevOps work items through the Azure DevOps API.
It provides functionality to create, update, and query work items.
*/
type WorkItem struct {
	conn *azuredevops.Connection
	wit  workitemtracking.Client
}

/*
NewWorkItem creates a new WorkItem instance with the provided Azure DevOps connection.

It initializes the work item tracking client for interacting with work items.
Returns nil if client initialization fails.
*/
func NewWorkItem(conn *azuredevops.Connection) *WorkItem {
	ctx := context.Background()
	wit, err := workitemtracking.NewClient(ctx, conn)

	if err != nil {
		errnie.Error(err)
		return nil
	}

	return &WorkItem{
		conn: conn,
		wit:  wit,
	}
}

/*
encode serializes the provided value into JSON and adds it to the artifact's payload.

Returns an error if JSON encoding fails.
*/
func (w *WorkItem) encode(artifact *datura.ArtifactBuilder, v any) (err error) {
	payload := bytes.NewBuffer([]byte{})

	if err = json.NewEncoder(payload).Encode(v); err != nil {
		return errnie.Error(err)
	}

	datura.WithPayload(payload.Bytes())(artifact)
	return nil
}

/*
CreateWorkItem creates a new work item in Azure DevOps.

It uses metadata from the artifact to set work item fields like title and description.
Returns an error if the creation fails.
*/
func (w *WorkItem) CreateWorkItem(artifact *datura.ArtifactBuilder) (err error) {
	ctx := context.Background()
	project := datura.GetMetaValue[string](artifact, "project")
	workItemType := datura.GetMetaValue[string](artifact, "type")

	operations := []webapi.JsonPatchOperation{
		{
			Op:    &webapi.OperationValues.Add,
			Path:  utils.Ptr("/fields/System.Title"),
			Value: datura.GetMetaValue[string](artifact, "title"),
		},
		{
			Op:    &webapi.OperationValues.Add,
			Path:  utils.Ptr("/fields/System.Description"),
			Value: datura.GetMetaValue[string](artifact, "description"),
		},
	}

	workItem, err := w.wit.CreateWorkItem(ctx, workitemtracking.CreateWorkItemArgs{
		Project:  &project,
		Type:     &workItemType,
		Document: &operations,
	})

	if err != nil {
		return errnie.Error(err)
	}

	return w.encode(artifact, workItem)
}

/*
UpdateWorkItem updates an existing work item in Azure DevOps.

It uses metadata from the artifact to update work item fields.
Returns an error if the update fails.
*/
func (w *WorkItem) UpdateWorkItem(artifact *datura.ArtifactBuilder) (err error) {
	ctx := context.Background()
	id := datura.GetMetaValue[int](artifact, "id")

	operations := []webapi.JsonPatchOperation{
		{
			Op:    &webapi.OperationValues.Replace,
			Path:  utils.Ptr("/fields/System.Title"),
			Value: datura.GetMetaValue[string](artifact, "title"),
		},
		{
			Op:    &webapi.OperationValues.Replace,
			Path:  utils.Ptr("/fields/System.Description"),
			Value: datura.GetMetaValue[string](artifact, "description"),
		},
	}

	workItem, err := w.wit.UpdateWorkItem(ctx, workitemtracking.UpdateWorkItemArgs{
		Id:       &id,
		Document: &operations,
	})

	if err != nil {
		return errnie.Error(err)
	}

	return w.encode(artifact, workItem)
}

/*
GetWorkItem retrieves a single work item from Azure DevOps by its ID.

The work item ID is extracted from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (w *WorkItem) GetWorkItem(artifact *datura.ArtifactBuilder) (err error) {
	ctx := context.Background()
	id := datura.GetMetaValue[int](artifact, "id")

	workItem, err := w.wit.GetWorkItem(ctx, workitemtracking.GetWorkItemArgs{
		Id: &id,
	})

	if err != nil {
		return errnie.Error(err)
	}

	return w.encode(artifact, workItem)
}

/*
ListWorkItems queries and retrieves multiple work items from Azure DevOps.

Uses a WIQL query from the artifact's metadata to filter work items.
Returns an error if the query fails.
*/
func (w *WorkItem) ListWorkItems(artifact *datura.ArtifactBuilder) (err error) {
	ctx := context.Background()
	project := datura.GetMetaValue[string](artifact, "project")
	query := datura.GetMetaValue[string](artifact, "query")

	wiql := workitemtracking.Wiql{
		Query: &query,
	}

	workItems, err := w.wit.QueryByWiql(ctx, workitemtracking.QueryByWiqlArgs{
		Project: &project,
		Wiql:    &wiql,
	})

	if err != nil {
		return errnie.Error(err)
	}

	return w.encode(artifact, workItems)
}
