package azure

import (
	"context"
	"fmt"

	"github.com/microsoft/azure-devops-go-api/azuredevops/v7"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7/webapi"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7/workitemtracking"
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

// Helper function to get string arguments
func getStringArg(args map[string]interface{}, key string) (string, error) {
	val, ok := args[key].(string)
	if !ok {
		return "", fmt.Errorf("missing or invalid type for argument '%s'", key)
	}
	return val, nil
}

// Helper function to get int arguments
func getIntArg(args map[string]interface{}, key string) (int, error) {
	// JSON numbers are often float64
	valFloat, ok := args[key].(float64)
	if !ok {
		// Try int directly just in case
		valInt, okInt := args[key].(int)
		if !okInt {
			return 0, fmt.Errorf("missing or invalid type for argument '%s'", key)
		}
		return valInt, nil
	}
	return int(valFloat), nil
}

/*
CreateWorkItem creates a new work item in Azure DevOps.

It uses arguments from the map to set work item fields like title and description.
Returns the created work item or an error.
*/
func (w *WorkItem) CreateWorkItem(ctx context.Context, args map[string]interface{}) (*workitemtracking.WorkItem, error) {
	project, err := getStringArg(args, "project")
	if err != nil {
		return nil, errnie.Error(err)
	}
	workItemType, err := getStringArg(args, "type")
	if err != nil {
		return nil, errnie.Error(err)
	}
	title, err := getStringArg(args, "title")
	if err != nil {
		return nil, errnie.Error(err)
	}
	description, err := getStringArg(args, "description")
	if err != nil {
		// Description might be optional, handle accordingly or make mandatory via schema
		description = "" // Default to empty if not provided/error
	}

	operations := []webapi.JsonPatchOperation{
		{
			Op:    &webapi.OperationValues.Add,
			Path:  utils.Ptr("/fields/System.Title"),
			Value: title,
		},
		{
			Op:    &webapi.OperationValues.Add,
			Path:  utils.Ptr("/fields/System.Description"),
			Value: description,
		},
	}

	workItem, err := w.wit.CreateWorkItem(ctx, workitemtracking.CreateWorkItemArgs{
		Project:  &project,
		Type:     &workItemType,
		Document: &operations,
	})

	if err != nil {
		return nil, errnie.Error(err)
	}

	return workItem, nil
}

/*
UpdateWorkItem updates an existing work item in Azure DevOps.

It uses arguments from the map to update work item fields.
Returns the updated work item or an error.
*/
func (w *WorkItem) UpdateWorkItem(ctx context.Context, args map[string]interface{}) (*workitemtracking.WorkItem, error) {
	id, err := getIntArg(args, "id")
	if err != nil {
		return nil, errnie.Error(err)
	}
	title, err := getStringArg(args, "title")
	if err != nil {
		return nil, errnie.Error(err) // Assume title is required for update
	}
	description, err := getStringArg(args, "description")
	if err != nil {
		description = "" // Default to empty if not provided/error
	}

	operations := []webapi.JsonPatchOperation{
		{
			Op:    &webapi.OperationValues.Replace,
			Path:  utils.Ptr("/fields/System.Title"),
			Value: title,
		},
		{
			Op:    &webapi.OperationValues.Replace,
			Path:  utils.Ptr("/fields/System.Description"),
			Value: description,
		},
	}

	workItem, err := w.wit.UpdateWorkItem(ctx, workitemtracking.UpdateWorkItemArgs{
		Id:       &id,
		Document: &operations,
	})

	if err != nil {
		return nil, errnie.Error(err)
	}

	return workItem, nil
}

/*
GetWorkItem retrieves a single work item from Azure DevOps by its ID.

The work item ID is extracted from the arguments map.
Returns the work item or an error.
*/
func (w *WorkItem) GetWorkItem(ctx context.Context, args map[string]interface{}) (*workitemtracking.WorkItem, error) {
	id, err := getIntArg(args, "id")
	if err != nil {
		return nil, errnie.Error(err)
	}

	workItem, err := w.wit.GetWorkItem(ctx, workitemtracking.GetWorkItemArgs{
		Id: &id,
	})

	if err != nil {
		return nil, errnie.Error(err)
	}

	return workItem, nil
}

/*
ListWorkItems queries and retrieves multiple work items from Azure DevOps.

Uses a WIQL query from the arguments map to filter work items.
Returns the query result or an error.
*/
func (w *WorkItem) ListWorkItems(ctx context.Context, args map[string]interface{}) (*workitemtracking.WorkItemQueryResult, error) {
	project, err := getStringArg(args, "project")
	if err != nil {
		return nil, errnie.Error(err)
	}
	query, err := getStringArg(args, "query")
	if err != nil {
		return nil, errnie.Error(err)
	}

	wiql := workitemtracking.Wiql{
		Query: &query,
	}

	workItems, err := w.wit.QueryByWiql(ctx, workitemtracking.QueryByWiqlArgs{
		Project: &project,
		Wiql:    &wiql,
	})

	if err != nil {
		return nil, errnie.Error(err)
	}

	return workItems, nil
}
