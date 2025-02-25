package examples

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/theapemachine/caramba/pkg/agent/tools"
)

// GitHubExample demonstrates the use of the GitHub tool
func GitHubExample() {
	// Get GitHub token from environment variable
	githubToken := os.Getenv("GITHUB_TOKEN")
	if githubToken == "" {
		fmt.Println("GITHUB_TOKEN environment variable not set")
		return
	}

	// Create a new GitHub tool with default owner and repo
	githubTool := tools.NewGitHubTool(githubToken, "theapemachine", "caramba")

	// Context for our operations
	ctx := context.Background()

	// Example 1: Search for code
	searchArgs := map[string]interface{}{
		"action": "search_code",
		"query":  "browser",
	}
	searchResult, err := githubTool.Execute(ctx, searchArgs)
	if err != nil {
		fmt.Printf("Error searching code: %v\n", err)
	} else {
		prettyPrint("Code Search Results", searchResult)
	}

	// Example 2: Get repository information
	repoArgs := map[string]interface{}{
		"action": "get_repo",
	}
	repoResult, err := githubTool.Execute(ctx, repoArgs)
	if err != nil {
		fmt.Printf("Error getting repository: %v\n", err)
	} else {
		prettyPrint("Repository Information", repoResult)
	}

	// Example 3: List open issues
	issuesArgs := map[string]interface{}{
		"action": "list_issues",
		"state":  "open",
	}
	issuesResult, err := githubTool.Execute(ctx, issuesArgs)
	if err != nil {
		fmt.Printf("Error listing issues: %v\n", err)
	} else {
		prettyPrint("Open Issues", issuesResult)
	}

	// Example 4: Get file content
	fileArgs := map[string]interface{}{
		"action": "get_file_content",
		"path":   "README.md",
	}
	fileResult, err := githubTool.Execute(ctx, fileArgs)
	if err != nil {
		fmt.Printf("Error getting file content: %v\n", err)
	} else {
		// Show only metadata, not the full content
		fileMap := fileResult.(map[string]interface{})
		fileMap["content"] = fmt.Sprintf("[%d bytes of content]", len(fileMap["content"].(string)))
		prettyPrint("README.md Metadata", fileMap)
	}

	// Example 5: List pull requests
	prArgs := map[string]interface{}{
		"action": "list_pull_requests",
		"state":  "all",
	}
	prResult, err := githubTool.Execute(ctx, prArgs)
	if err != nil {
		fmt.Printf("Error listing pull requests: %v\n", err)
	} else {
		prettyPrint("Pull Requests", prResult)
	}

	// Note: The following examples are commented out as they would create/modify actual resources
	// Uncomment and modify them if you want to test these operations

	/*
		// Example 6: Create an issue (disabled by default)
		issueArgs := map[string]interface{}{
			"action": "create_issue",
			"title":  "Test Issue from Caramba",
			"body":   "This is a test issue created by the Caramba GitHub tool example.",
		}
		issueResult, err := githubTool.Execute(ctx, issueArgs)
		if err != nil {
			fmt.Printf("Error creating issue: %v\n", err)
		} else {
			prettyPrint("Created Issue", issueResult)
		}

		// Example 7: Create a pull request (disabled by default)
		prCreateArgs := map[string]interface{}{
			"action": "create_pull_request",
			"title":  "Test PR from Caramba",
			"body":   "This is a test pull request created by the Caramba GitHub tool example.",
			"head":   "feature-branch", // Replace with actual branch name
			"base":   "main",           // Replace with actual branch name
		}
		prCreateResult, err := githubTool.Execute(ctx, prCreateArgs)
		if err != nil {
			fmt.Printf("Error creating pull request: %v\n", err)
		} else {
			prettyPrint("Created Pull Request", prCreateResult)
		}
	*/
}

// prettyPrint formats and prints the result with a title
func prettyPrint(title string, data interface{}) {
	fmt.Printf("\n--- %s ---\n", title)
	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		fmt.Printf("Error formatting JSON: %v\n", err)
		return
	}
	fmt.Println(string(jsonData))
	fmt.Println()
}
