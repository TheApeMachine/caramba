package tools

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"github.com/google/uuid"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7/core"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7/webapi"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7/work"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7/workitemtracking"
)

// AzureDevOpsTool provides integration with the Azure DevOps API
type AzureDevOpsTool struct {
	// client is the Azure DevOps connection
	connection *azuredevops.Connection
	// organization is the Azure DevOps organization
	organization string
	// project is the default Azure DevOps project
	project string
	// personalAccessToken is the PAT for authentication
	personalAccessToken string
}

// NewAzureDevOpsTool creates a new AzureDevOpsTool
func NewAzureDevOpsTool(organization, personalAccessToken, defaultProject string) *AzureDevOpsTool {
	// Create Azure DevOps connection
	orgURL := fmt.Sprintf("https://dev.azure.com/%s", organization)
	connection := azuredevops.NewPatConnection(orgURL, personalAccessToken)

	return &AzureDevOpsTool{
		connection:          connection,
		organization:        organization,
		project:             defaultProject,
		personalAccessToken: personalAccessToken,
	}
}

// Name returns the name of the tool
func (a *AzureDevOpsTool) Name() string {
	return "azure_devops"
}

// Description returns the description of the tool
func (a *AzureDevOpsTool) Description() string {
	return "Integrates with Azure DevOps API for work items, boards, repositories, and more"
}

// Execute executes the tool with the given arguments
func (a *AzureDevOpsTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	action, ok := args["action"].(string)
	if !ok {
		return nil, errors.New("action must be a string")
	}

	switch action {
	case "get_work_item":
		return a.getWorkItem(ctx, args)
	case "create_work_item":
		return a.createWorkItem(ctx, args)
	case "update_work_item":
		return a.updateWorkItem(ctx, args)
	case "list_work_items":
		return a.listWorkItems(ctx, args)
	case "get_project":
		return a.getProject(ctx, args)
	case "list_projects":
		return a.listProjects(ctx, args)
	case "get_team":
		return a.getTeam(ctx, args)
	case "list_teams":
		return a.listTeams(ctx, args)
	case "get_sprint":
		return a.getSprint(ctx, args)
	case "list_sprints":
		return a.listSprints(ctx, args)
	case "query_work_items":
		return a.queryWorkItems(ctx, args)
	case "create_comment":
		return a.createComment(ctx, args)
	case "get_comments":
		return a.getComments(ctx, args)
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

// Schema returns the JSON schema for the tool's arguments
func (a *AzureDevOpsTool) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"action": map[string]interface{}{
				"type": "string",
				"enum": []string{
					"get_work_item",
					"create_work_item",
					"update_work_item",
					"list_work_items",
					"get_project",
					"list_projects",
					"get_team",
					"list_teams",
					"get_sprint",
					"list_sprints",
					"query_work_items",
					"create_comment",
					"get_comments",
				},
				"description": "Action to perform with the Azure DevOps API",
			},
			"project": map[string]interface{}{
				"type":        "string",
				"description": "Azure DevOps project name",
			},
			"id": map[string]interface{}{
				"type":        "number",
				"description": "Work item ID (for operations on a specific work item)",
			},
			"work_item_type": map[string]interface{}{
				"type": "string",
				"enum": []string{
					"Epic",
					"Feature",
					"User Story",
					"Bug",
					"Task",
				},
				"description": "Type of work item (for creation)",
			},
			"title": map[string]interface{}{
				"type":        "string",
				"description": "Title for the work item",
			},
			"description": map[string]interface{}{
				"type":        "string",
				"description": "Description content for the work item",
			},
			"state": map[string]interface{}{
				"type":        "string",
				"description": "State for the work item (e.g., New, Active, Closed)",
			},
			"priority": map[string]interface{}{
				"type":        "number",
				"description": "Priority for the work item (1-4)",
			},
			"team": map[string]interface{}{
				"type":        "string",
				"description": "Team name",
			},
			"query": map[string]interface{}{
				"type":        "string",
				"description": "WIQL query to find work items",
			},
			"comment": map[string]interface{}{
				"type":        "string",
				"description": "Text for a comment",
			},
		},
		"required": []string{"action"},
	}
}

// Helper functions for converting types to pointers
func stringPtr(s string) *string {
	return &s
}

func boolPtr(b bool) *bool {
	return &b
}

func intPtr(i int) *int {
	return &i
}

// Helper function to convert Operation to a pointer
func operationPtr(op webapi.Operation) *webapi.Operation {
	return &op
}

// getProject gets the project with specified arguments or default
func (a *AzureDevOpsTool) getProject(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the project client
	projectClient, err := core.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create project client: %w", err)
	}

	// Determine the project name
	projectName, ok := args["project"].(string)
	if !ok || projectName == "" {
		if a.project == "" {
			return nil, errors.New("project must be specified")
		}
		projectName = a.project
	}

	// Get project details
	project, err := projectClient.GetProject(ctx, core.GetProjectArgs{
		ProjectId:           &projectName,
		IncludeCapabilities: boolPtr(true),
		IncludeHistory:      boolPtr(false),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get project: %w", err)
	}

	return map[string]interface{}{
		"id":          project.Id.String(),
		"name":        *project.Name,
		"description": getStringOrDefault(project.Description, ""),
		"url":         *project.Url,
		"state":       string(*project.State),
		"revision":    *project.Revision,
		"visibility":  string(*project.Visibility),
	}, nil
}

// listProjects lists all projects
func (a *AzureDevOpsTool) listProjects(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the project client
	projectClient, err := core.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create project client: %w", err)
	}

	// List all projects
	projectList, err := projectClient.GetProjects(ctx, core.GetProjectsArgs{
		StateFilter: func() *core.ProjectState {
			state := core.ProjectStateValues.All
			return &state
		}(),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list projects: %w", err)
	}

	// Convert to a simpler format
	result := make([]map[string]interface{}, 0)
	if projectList != nil && projectList.Value != nil {
		for _, project := range projectList.Value {
			result = append(result, map[string]interface{}{
				"id":          project.Id.String(),
				"name":        *project.Name,
				"description": getStringOrDefault(project.Description, ""),
				"url":         *project.Url,
				"state":       string(*project.State),
			})
		}
	}

	return map[string]interface{}{
		"count":    len(result),
		"projects": result,
	}, nil
}

// getTeam gets a specific team
func (a *AzureDevOpsTool) getTeam(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the core client for teams
	coreClient, err := core.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create core client: %w", err)
	}

	// Determine the project name
	projectName, ok := args["project"].(string)
	if !ok || projectName == "" {
		if a.project == "" {
			return nil, errors.New("project must be specified")
		}
		projectName = a.project
	}

	// Determine the team name
	teamName, ok := args["team"].(string)
	if !ok || teamName == "" {
		return nil, errors.New("team name must be specified")
	}

	// Get team details
	team, err := coreClient.GetTeam(ctx, core.GetTeamArgs{
		ProjectId: &projectName,
		TeamId:    &teamName,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get team: %w", err)
	}

	return map[string]interface{}{
		"id":          team.Id.String(),
		"name":        *team.Name,
		"description": getStringOrDefault(team.Description, ""),
		"url":         *team.Url,
	}, nil
}

// listTeams lists all teams in a project
func (a *AzureDevOpsTool) listTeams(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the core client for teams
	coreClient, err := core.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create core client: %w", err)
	}

	// Determine the project name
	projectName, ok := args["project"].(string)
	if !ok || projectName == "" {
		if a.project == "" {
			return nil, errors.New("project must be specified")
		}
		projectName = a.project
	}

	// List all teams in the project
	teams, err := coreClient.GetTeams(ctx, core.GetTeamsArgs{
		ProjectId: &projectName,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list teams: %w", err)
	}

	// Convert to a simpler format
	result := make([]map[string]interface{}, 0, len(*teams))
	for _, team := range *teams {
		result = append(result, map[string]interface{}{
			"id":          team.Id.String(),
			"name":        *team.Name,
			"description": getStringOrDefault(team.Description, ""),
			"url":         *team.Url,
		})
	}

	return map[string]interface{}{
		"count": len(*teams),
		"teams": result,
	}, nil
}

// getSprint gets a specific sprint
func (a *AzureDevOpsTool) getSprint(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the work client
	workClient, err := work.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create work client: %w", err)
	}

	// Determine the project name
	projectName, ok := args["project"].(string)
	if !ok || projectName == "" {
		if a.project == "" {
			return nil, errors.New("project must be specified")
		}
		projectName = a.project
	}

	// Determine the team name
	teamName, ok := args["team"].(string)
	if !ok || teamName == "" {
		return nil, errors.New("team name must be specified")
	}

	// Determine the sprint ID
	sprintID, ok := args["id"].(float64)
	if !ok {
		return nil, errors.New("sprint ID must be specified")
	}

	// Convert to string
	sprintIDStr := strconv.Itoa(int(sprintID))

	// Parse the string as UUID
	sprintUUID, err := uuid.Parse(sprintIDStr)
	if err != nil {
		return nil, fmt.Errorf("failed to parse sprint ID as UUID: %w", err)
	}

	// Get sprint details
	sprint, err := workClient.GetTeamIteration(ctx, work.GetTeamIterationArgs{
		Project: &projectName,
		Team:    &teamName,
		Id:      &sprintUUID,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get sprint: %w", err)
	}

	var startDate, finishDate string
	if sprint.Attributes != nil {
		if sprint.Attributes.StartDate != nil {
			startDate = sprint.Attributes.StartDate.String()
		}
		if sprint.Attributes.FinishDate != nil {
			finishDate = sprint.Attributes.FinishDate.String()
		}
	}

	return map[string]interface{}{
		"id":          *sprint.Id,
		"name":        *sprint.Name,
		"path":        *sprint.Path,
		"url":         *sprint.Url,
		"start_date":  startDate,
		"finish_date": finishDate,
	}, nil
}

// listSprints lists all sprints for a team
func (a *AzureDevOpsTool) listSprints(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the work client
	workClient, err := work.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create work client: %w", err)
	}

	// Determine the project name
	projectName, ok := args["project"].(string)
	if !ok || projectName == "" {
		if a.project == "" {
			return nil, errors.New("project must be specified")
		}
		projectName = a.project
	}

	// Determine the team name
	teamName, ok := args["team"].(string)
	if !ok || teamName == "" {
		return nil, errors.New("team name must be specified")
	}

	// List all sprints for the team
	sprints, err := workClient.GetTeamIterations(ctx, work.GetTeamIterationsArgs{
		Project: &projectName,
		Team:    &teamName,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list sprints: %w", err)
	}

	// Convert to a simpler format
	result := make([]map[string]interface{}, 0, len(*sprints))
	for _, sprint := range *sprints {
		var startDate, finishDate string
		if sprint.Attributes != nil {
			if sprint.Attributes.StartDate != nil {
				startDate = sprint.Attributes.StartDate.String()
			}
			if sprint.Attributes.FinishDate != nil {
				finishDate = sprint.Attributes.FinishDate.String()
			}
		}

		result = append(result, map[string]interface{}{
			"id":          *sprint.Id,
			"name":        *sprint.Name,
			"path":        *sprint.Path,
			"url":         *sprint.Url,
			"start_date":  startDate,
			"finish_date": finishDate,
		})
	}

	return map[string]interface{}{
		"count":   len(*sprints),
		"sprints": result,
	}, nil
}

// getWorkItem gets a specific work item
func (a *AzureDevOpsTool) getWorkItem(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the work item tracking client
	witClient, err := workitemtracking.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create work item tracking client: %w", err)
	}

	// Determine the work item ID
	id, ok := args["id"].(float64)
	if !ok {
		return nil, errors.New("work item ID must be specified")
	}

	// Convert to int and create a pointer
	idInt := int(id)

	// Get work item details
	workItem, err := witClient.GetWorkItem(ctx, workitemtracking.GetWorkItemArgs{
		Id:     &idInt,
		Expand: &workitemtracking.WorkItemExpandValues.All,
		AsOf:   nil,
		Fields: nil,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get work item: %w", err)
	}

	// Extract relevant fields
	result := map[string]interface{}{
		"id":   *workItem.Id,
		"url":  *workItem.Url,
		"rev":  *workItem.Rev,
		"type": nil,
	}

	// Add fields
	if workItem.Fields != nil {
		for key, value := range *workItem.Fields {
			result[fieldNameToKey(key)] = value
		}
	}

	return result, nil
}

// createWorkItem creates a new work item
func (a *AzureDevOpsTool) createWorkItem(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the work item tracking client
	witClient, err := workitemtracking.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create work item tracking client: %w", err)
	}

	// Determine the project name
	projectName, ok := args["project"].(string)
	if !ok || projectName == "" {
		if a.project == "" {
			return nil, errors.New("project must be specified")
		}
		projectName = a.project
	}

	// Work item type
	workItemType, ok := args["work_item_type"].(string)
	if !ok || workItemType == "" {
		return nil, errors.New("work_item_type must be specified")
	}

	// Title
	title, ok := args["title"].(string)
	if !ok || title == "" {
		return nil, errors.New("title must be specified")
	}

	// Description (optional)
	description, _ := args["description"].(string)

	// Prepare the document
	jsonPatchOps := []webapi.JsonPatchOperation{
		{
			Op:    operationPtr(webapi.OperationValues.Add),
			Path:  stringPtr("/fields/System.Title"),
			Value: title,
		},
	}

	// Add description if provided
	if description != "" {
		jsonPatchOps = append(jsonPatchOps, webapi.JsonPatchOperation{
			Op:    operationPtr(webapi.OperationValues.Add),
			Path:  stringPtr("/fields/System.Description"),
			Value: description,
		})
	}

	// Add priority if provided
	if priority, ok := args["priority"].(float64); ok {
		jsonPatchOps = append(jsonPatchOps, webapi.JsonPatchOperation{
			Op:    operationPtr(webapi.OperationValues.Add),
			Path:  stringPtr("/fields/Microsoft.VSTS.Common.Priority"),
			Value: int(priority),
		})
	}

	// Create the work item
	workItem, err := witClient.CreateWorkItem(ctx, workitemtracking.CreateWorkItemArgs{
		Document:              &jsonPatchOps,
		Project:               &projectName,
		Type:                  &workItemType,
		ValidateOnly:          nil,
		BypassRules:           nil,
		SuppressNotifications: nil,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create work item: %w", err)
	}

	return map[string]interface{}{
		"id":   *workItem.Id,
		"url":  *workItem.Url,
		"rev":  *workItem.Rev,
		"type": workItemType,
	}, nil
}

// updateWorkItem updates an existing work item
func (a *AzureDevOpsTool) updateWorkItem(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the work item tracking client
	witClient, err := workitemtracking.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create work item tracking client: %w", err)
	}

	// Determine the work item ID
	id, ok := args["id"].(float64)
	if !ok {
		return nil, errors.New("work item ID must be specified")
	}

	// Convert to int and create a pointer
	idInt := int(id)

	// Prepare the document
	jsonPatchOps := []webapi.JsonPatchOperation{}

	// Title (optional)
	if title, ok := args["title"].(string); ok && title != "" {
		jsonPatchOps = append(jsonPatchOps, webapi.JsonPatchOperation{
			Op:    operationPtr(webapi.OperationValues.Add),
			Path:  stringPtr("/fields/System.Title"),
			Value: title,
		})
	}

	// Description (optional)
	if description, ok := args["description"].(string); ok && description != "" {
		jsonPatchOps = append(jsonPatchOps, webapi.JsonPatchOperation{
			Op:    operationPtr(webapi.OperationValues.Add),
			Path:  stringPtr("/fields/System.Description"),
			Value: description,
		})
	}

	// State (optional)
	if state, ok := args["state"].(string); ok && state != "" {
		jsonPatchOps = append(jsonPatchOps, webapi.JsonPatchOperation{
			Op:    operationPtr(webapi.OperationValues.Add),
			Path:  stringPtr("/fields/System.State"),
			Value: state,
		})
	}

	// Priority (optional)
	if priority, ok := args["priority"].(float64); ok {
		jsonPatchOps = append(jsonPatchOps, webapi.JsonPatchOperation{
			Op:    operationPtr(webapi.OperationValues.Add),
			Path:  stringPtr("/fields/Microsoft.VSTS.Common.Priority"),
			Value: int(priority),
		})
	}

	// Check if we have any operations
	if len(jsonPatchOps) == 0 {
		return nil, errors.New("at least one field to update must be specified")
	}

	// Update the work item
	workItem, err := witClient.UpdateWorkItem(ctx, workitemtracking.UpdateWorkItemArgs{
		Document:              &jsonPatchOps,
		Id:                    &idInt,
		ValidateOnly:          nil,
		BypassRules:           nil,
		SuppressNotifications: nil,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to update work item: %w", err)
	}

	return map[string]interface{}{
		"id":  *workItem.Id,
		"url": *workItem.Url,
		"rev": *workItem.Rev,
	}, nil
}

// listWorkItems lists work items
func (a *AzureDevOpsTool) listWorkItems(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Since Azure DevOps doesn't have a direct "list all work items" API,
	// we'll use a WIQL query to get work items
	return a.queryWorkItems(ctx, map[string]interface{}{
		"query":   "SELECT [System.Id], [System.Title], [System.WorkItemType], [System.State], [System.AssignedTo], [System.CreatedDate] FROM WorkItems WHERE [System.TeamProject] = @project ORDER BY [System.CreatedDate] DESC",
		"project": args["project"],
	})
}

// queryWorkItems executes a WIQL query to find work items
func (a *AzureDevOpsTool) queryWorkItems(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the work item tracking client
	witClient, err := workitemtracking.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create work item tracking client: %w", err)
	}

	// Determine the project name
	projectName, ok := args["project"].(string)
	if !ok || projectName == "" {
		if a.project == "" {
			return nil, errors.New("project must be specified")
		}
		projectName = a.project
	}

	// Get the WIQL query
	query, ok := args["query"].(string)
	if !ok || query == "" {
		return nil, errors.New("query must be specified")
	}

	// Execute the query
	wiqlQuery := workitemtracking.Wiql{
		Query: stringPtr(query),
	}

	queryResult, err := witClient.QueryByWiql(ctx, workitemtracking.QueryByWiqlArgs{
		Wiql:          &wiqlQuery,
		Project:       &projectName,
		Team:          nil,
		Top:           intPtr(50),
		TimePrecision: nil,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}

	// If no work items found
	if queryResult.WorkItems == nil || len(*queryResult.WorkItems) == 0 {
		return map[string]interface{}{
			"count":      0,
			"work_items": []interface{}{},
		}, nil
	}

	// Extract work item IDs
	ids := make([]int, 0, len(*queryResult.WorkItems))
	for _, wi := range *queryResult.WorkItems {
		ids = append(ids, *wi.Id)
	}

	// Get work item details for the found IDs
	workItems, err := witClient.GetWorkItems(ctx, workitemtracking.GetWorkItemsArgs{
		Ids:    &ids,
		Fields: nil,
		AsOf:   nil,
		Expand: &workitemtracking.WorkItemExpandValues.None,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get work item details: %w", err)
	}

	// Convert to a simpler format
	result := make([]map[string]interface{}, 0, len(*workItems))
	for _, wi := range *workItems {
		item := map[string]interface{}{
			"id":  *wi.Id,
			"url": *wi.Url,
			"rev": *wi.Rev,
		}

		// Add fields
		if wi.Fields != nil {
			for key, value := range *wi.Fields {
				item[fieldNameToKey(key)] = value
			}
		}

		result = append(result, item)
	}

	return map[string]interface{}{
		"count":      len(*workItems),
		"work_items": result,
	}, nil
}

// createComment adds a comment to a work item
func (a *AzureDevOpsTool) createComment(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the work item tracking client
	witClient, err := workitemtracking.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create work item tracking client: %w", err)
	}

	// Determine the work item ID
	id, ok := args["id"].(float64)
	if !ok {
		return nil, errors.New("work item ID must be specified")
	}

	// Convert to int and create a pointer
	idInt := int(id)

	// Get the comment text
	comment, ok := args["comment"].(string)
	if !ok || comment == "" {
		return nil, errors.New("comment must be specified")
	}

	// Create the comment
	commentRequest := workitemtracking.CommentCreate{
		Text: stringPtr(comment),
	}

	commentResult, err := witClient.AddComment(ctx, workitemtracking.AddCommentArgs{
		Request:    &commentRequest,
		WorkItemId: &idInt,
		Project:    nil,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to add comment: %w", err)
	}

	return map[string]interface{}{
		"id":         *commentResult.Id,
		"text":       *commentResult.Text,
		"created_by": *commentResult.CreatedBy.DisplayName,
		"created_at": commentResult.CreatedDate.String(),
		"url":        *commentResult.Url,
	}, nil
}

// getComments gets comments for a work item
func (a *AzureDevOpsTool) getComments(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the work item tracking client
	witClient, err := workitemtracking.NewClient(ctx, a.connection)
	if err != nil {
		return nil, fmt.Errorf("failed to create work item tracking client: %w", err)
	}

	// Determine the work item ID
	id, ok := args["id"].(float64)
	if !ok {
		return nil, errors.New("work item ID must be specified")
	}

	// Convert to int and create a pointer
	idInt := int(id)

	// Get the comments
	comments, err := witClient.GetComments(ctx, workitemtracking.GetCommentsArgs{
		WorkItemId: &idInt,
		Project:    nil,
		Top:        intPtr(100),
		Expand:     nil,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get comments: %w", err)
	}

	// Convert to a simpler format
	result := make([]map[string]interface{}, 0, len(*comments.Comments))
	for _, comment := range *comments.Comments {
		result = append(result, map[string]interface{}{
			"id":         *comment.Id,
			"text":       *comment.Text,
			"created_by": *comment.CreatedBy.DisplayName,
			"created_at": comment.CreatedDate.String(),
			"url":        *comment.Url,
		})
	}

	return map[string]interface{}{
		"count":    *comments.TotalCount,
		"comments": result,
	}, nil
}

// Helper functions

// getStringOrDefault returns the string value or default if nil
func getStringOrDefault(value *string, defaultValue string) string {
	if value == nil {
		return defaultValue
	}
	return *value
}

// fieldNameToKey converts a field name like "System.Title" to a simpler key like "title"
func fieldNameToKey(fieldName string) string {
	switch fieldName {
	case "System.Title":
		return "title"
	case "System.Description":
		return "description"
	case "System.State":
		return "state"
	case "System.WorkItemType":
		return "type"
	case "System.AssignedTo":
		return "assigned_to"
	case "System.CreatedDate":
		return "created_date"
	case "System.CreatedBy":
		return "created_by"
	case "System.ChangedDate":
		return "changed_date"
	case "System.ChangedBy":
		return "changed_by"
	case "Microsoft.VSTS.Common.Priority":
		return "priority"
	default:
		return fieldName
	}
}
