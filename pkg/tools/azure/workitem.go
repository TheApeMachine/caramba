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

type WorkItem struct {
	conn *azuredevops.Connection
	wit  workitemtracking.Client
}

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

func (w *WorkItem) encode(artifact *datura.Artifact, v any) (err error) {
	payload := bytes.NewBuffer([]byte{})

	if err = json.NewEncoder(payload).Encode(v); err != nil {
		return errnie.Error(err)
	}

	datura.WithPayload(payload.Bytes())(artifact)
	return nil
}

func (w *WorkItem) CreateWorkItem(artifact *datura.Artifact) (err error) {
	ctx := context.Background()
	project := datura.GetMetaValue[string](artifact, "project")
	workItemType := datura.GetMetaValue[string](artifact, "type")

	operations := []webapi.JsonPatchOperation{
		{
			Op:    &webapi.OperationValues.Add,
			Path:  utils.Ptr[string]("/fields/System.Title"),
			Value: datura.GetMetaValue[string](artifact, "title"),
		},
		{
			Op:    &webapi.OperationValues.Add,
			Path:  utils.Ptr[string]("/fields/System.Description"),
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

func (w *WorkItem) UpdateWorkItem(artifact *datura.Artifact) (err error) {
	ctx := context.Background()
	id := datura.GetMetaValue[int](artifact, "id")

	operations := []webapi.JsonPatchOperation{
		{
			Op:    &webapi.OperationValues.Replace,
			Path:  utils.Ptr[string]("/fields/System.Title"),
			Value: datura.GetMetaValue[string](artifact, "title"),
		},
		{
			Op:    &webapi.OperationValues.Replace,
			Path:  utils.Ptr[string]("/fields/System.Description"),
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

func (w *WorkItem) GetWorkItem(artifact *datura.Artifact) (err error) {
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

func (w *WorkItem) ListWorkItems(artifact *datura.Artifact) (err error) {
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

func convertPtr(s string) *string {
	return &s
}
