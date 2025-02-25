package examples

import (
	"context"
	"fmt"
	"os"

	"github.com/theapemachine/caramba/pkg/agent/tools"
)

// AzureDevOpsExample demonstrates the use of the Azure DevOps tool
func AzureDevOpsExample() {
	// Get Azure DevOps credentials from environment variables
	organization := os.Getenv("AZURE_DEVOPS_ORG")
	if organization == "" {
		fmt.Println("AZURE_DEVOPS_ORG environment variable not set")
		return
	}

	pat := os.Getenv("AZURE_DEVOPS_PAT")
	if pat == "" {
		fmt.Println("AZURE_DEVOPS_PAT environment variable not set")
		return
	}

	project := os.Getenv("AZURE_DEVOPS_PROJECT")
	if project == "" {
		fmt.Println("AZURE_DEVOPS_PROJECT environment variable not set")
		return
	}

	// Create a new Azure DevOps tool
	azureDevOpsTool := tools.NewAzureDevOpsTool(organization, pat, project)

	// Context for our operations
	ctx := context.Background()

	// Example 1: Get project information
	projectArgs := map[string]interface{}{
		"action": "get_project",
	}
	projectResult, err := azureDevOpsTool.Execute(ctx, projectArgs)
	if err != nil {
		fmt.Printf("Error getting project: %v\n", err)
	} else {
		prettyPrint("Project Information", projectResult)
	}

	// Example 2: List teams in the project
	teamsArgs := map[string]interface{}{
		"action": "list_teams",
	}
	teamsResult, err := azureDevOpsTool.Execute(ctx, teamsArgs)
	if err != nil {
		fmt.Printf("Error listing teams: %v\n", err)
	} else {
		prettyPrint("Teams", teamsResult)
	}

	// Example 3: Query work items
	queryArgs := map[string]interface{}{
		"action": "query_work_items",
		"query":  "SELECT [System.Id], [System.Title], [System.WorkItemType], [System.State] FROM WorkItems WHERE [System.TeamProject] = @project AND [System.WorkItemType] = 'User Story' ORDER BY [System.ChangedDate] DESC",
	}
	queryResult, err := azureDevOpsTool.Execute(ctx, queryArgs)
	if err != nil {
		fmt.Printf("Error querying work items: %v\n", err)
	} else {
		prettyPrint("User Stories", queryResult)
	}

	// Note: The following examples are commented out as they would create/modify actual resources
	// Uncomment and modify them if you want to test these operations

	/*
		// Example 4: Create a work item (disabled by default)
		createArgs := map[string]interface{}{
			"action":         "create_work_item",
			"work_item_type": "User Story",
			"title":          "Test User Story from Caramba",
			"description":    "This is a test user story created by the Caramba Azure DevOps tool example.",
			"priority":       2,
		}
		createResult, err := azureDevOpsTool.Execute(ctx, createArgs)
		if err != nil {
			fmt.Printf("Error creating work item: %v\n", err)
		} else {
			prettyPrint("Created Work Item", createResult)

			// Example 5: Add a comment to the work item (disabled by default)
			workItemID := createResult.(map[string]interface{})["id"].(int)
			commentArgs := map[string]interface{}{
				"action":  "create_comment",
				"id":      float64(workItemID),
				"comment": "This is a test comment added by the Caramba Azure DevOps tool example.",
			}
			commentResult, err := azureDevOpsTool.Execute(ctx, commentArgs)
			if err != nil {
				fmt.Printf("Error adding comment: %v\n", err)
			} else {
				prettyPrint("Added Comment", commentResult)
			}

			// Example 6: Update the work item (disabled by default)
			updateArgs := map[string]interface{}{
				"action": "update_work_item",
				"id":     float64(workItemID),
				"title":  "Updated User Story from Caramba",
				"state":  "Active",
			}
			updateResult, err := azureDevOpsTool.Execute(ctx, updateArgs)
			if err != nil {
				fmt.Printf("Error updating work item: %v\n", err)
			} else {
				prettyPrint("Updated Work Item", updateResult)
			}
		}
	*/
}
