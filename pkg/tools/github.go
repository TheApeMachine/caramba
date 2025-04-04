package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
)

/* GithubTool provides a base for all GitHub operations */
type GithubTool struct {
	operations map[string]ToolType
}

/* NewGithubTool creates a new GitHub tool with all operations */
func NewGithubTool() *GithubTool {
	getRepositories := NewGithubGetRepositoriesTool()
	getRepository := NewGithubGetRepositoryTool()
	createRepository := NewGithubCreateRepositoryTool()
	deleteRepository := NewGithubDeleteRepositoryTool()
	updateRepository := NewGithubUpdateRepositoryTool()
	listIssues := NewGithubListIssuesTool()
	createIssue := NewGithubCreateIssueTool()
	updateIssue := NewGithubUpdateIssueTool()
	closeIssue := NewGithubCloseIssueTool()
	listPulls := NewGithubListPullsTool()
	createPull := NewGithubCreatePullTool()
	updatePull := NewGithubUpdatePullTool()
	mergePull := NewGithubMergePullTool()
	listReviews := NewGithubListReviewsTool()
	createReview := NewGithubCreateReviewTool()
	updateReview := NewGithubUpdateReviewTool()
	listReviewComments := NewGithubListReviewCommentsTool()
	createReviewComment := NewGithubCreateReviewCommentTool()

	return &GithubTool{
		operations: map[string]ToolType{
			"get_repositories":      {getRepositories.Tool, getRepositories.Use},
			"get_repository":        {getRepository.Tool, getRepository.Use},
			"create_repository":     {createRepository.Tool, createRepository.Use},
			"delete_repository":     {deleteRepository.Tool, deleteRepository.Use},
			"update_repository":     {updateRepository.Tool, updateRepository.Use},
			"list_issues":           {listIssues.Tool, listIssues.Use},
			"create_issue":          {createIssue.Tool, createIssue.Use},
			"update_issue":          {updateIssue.Tool, updateIssue.Use},
			"close_issue":           {closeIssue.Tool, closeIssue.Use},
			"list_pulls":            {listPulls.Tool, listPulls.Use},
			"create_pull":           {createPull.Tool, createPull.Use},
			"update_pull":           {updatePull.Tool, updatePull.Use},
			"merge_pull":            {mergePull.Tool, mergePull.Use},
			"list_reviews":          {listReviews.Tool, listReviews.Use},
			"create_review":         {createReview.Tool, createReview.Use},
			"update_review":         {updateReview.Tool, updateReview.Use},
			"list_review_comments":  {listReviewComments.Tool, listReviewComments.Use},
			"create_review_comment": {createReviewComment.Tool, createReviewComment.Use},
		},
	}
}

/* ToMCP returns all GitHub tool definitions */
func (tool *GithubTool) ToMCP() []ToolType {
	tools := make([]ToolType, 0)

	for _, tool := range tool.operations {
		tools = append(tools, tool)
	}

	return tools
}

/* GithubGetRepositoriesTool implements a tool for getting repositories from GitHub */
type GithubGetRepositoriesTool struct {
	mcp.Tool
}

/* NewGithubGetRepositoriesTool creates a new tool for getting repositories */
func NewGithubGetRepositoriesTool() *GithubGetRepositoriesTool {
	return &GithubGetRepositoriesTool{
		Tool: mcp.NewTool(
			"get_repositories",
			mcp.WithDescription("A tool for getting repositories from GitHub."),
		),
	}
}

/* Use executes the get repositories operation and returns the results */
func (tool *GithubGetRepositoriesTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubGetRepositoryTool implements a tool for getting a repository from GitHub */
type GithubGetRepositoryTool struct {
	mcp.Tool
}

/* NewGithubGetRepositoryTool creates a new tool for getting a repository */
func NewGithubGetRepositoryTool() *GithubGetRepositoryTool {
	return &GithubGetRepositoryTool{
		Tool: mcp.NewTool(
			"get_repository",
			mcp.WithDescription("A tool for getting a repository from GitHub."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository to get."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the get repository operation and returns the results */
func (tool *GithubGetRepositoryTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubCreateRepositoryTool implements a tool for creating a repository */
type GithubCreateRepositoryTool struct {
	mcp.Tool
}

/* NewGithubCreateRepositoryTool creates a new tool for creating repositories */
func NewGithubCreateRepositoryTool() *GithubCreateRepositoryTool {
	return &GithubCreateRepositoryTool{
		Tool: mcp.NewTool(
			"create_repository",
			mcp.WithDescription("A tool for creating a repository on GitHub."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository to create."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the create repository operation and returns the results */
func (tool *GithubCreateRepositoryTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubDeleteRepositoryTool implements a tool for deleting a repository */
type GithubDeleteRepositoryTool struct {
	mcp.Tool
}

/* NewGithubDeleteRepositoryTool creates a new tool for deleting repositories */
func NewGithubDeleteRepositoryTool() *GithubDeleteRepositoryTool {
	return &GithubDeleteRepositoryTool{
		Tool: mcp.NewTool(
			"delete_repository",
			mcp.WithDescription("A tool for deleting a repository on GitHub."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository to delete."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the delete repository operation */
func (tool *GithubDeleteRepositoryTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubUpdateRepositoryTool implements a tool for updating a repository */
type GithubUpdateRepositoryTool struct {
	mcp.Tool
}

/* NewGithubUpdateRepositoryTool creates a new tool for updating repositories */
func NewGithubUpdateRepositoryTool() *GithubUpdateRepositoryTool {
	return &GithubUpdateRepositoryTool{
		Tool: mcp.NewTool(
			"update_repository",
			mcp.WithDescription("A tool for updating a repository on GitHub."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository to update."),
				mcp.Required(),
			),
			mcp.WithString(
				"description",
				mcp.Description("New description for the repository."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the update repository operation */
func (tool *GithubUpdateRepositoryTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubListIssuesTool implements a tool for listing repository issues */
type GithubListIssuesTool struct {
	mcp.Tool
}

/* NewGithubListIssuesTool creates a new tool for listing issues */
func NewGithubListIssuesTool() *GithubListIssuesTool {
	return &GithubListIssuesTool{
		Tool: mcp.NewTool(
			"list_issues",
			mcp.WithDescription("A tool for listing issues in a GitHub repository."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository to list issues from."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the list issues operation */
func (tool *GithubListIssuesTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubCreateIssueTool implements a tool for creating an issue */
type GithubCreateIssueTool struct {
	mcp.Tool
}

/* NewGithubCreateIssueTool creates a new tool for creating issues */
func NewGithubCreateIssueTool() *GithubCreateIssueTool {
	return &GithubCreateIssueTool{
		Tool: mcp.NewTool(
			"create_issue",
			mcp.WithDescription("A tool for creating an issue in a GitHub repository."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository to create the issue in."),
				mcp.Required(),
			),
			mcp.WithString(
				"title",
				mcp.Description("The title of the issue."),
				mcp.Required(),
			),
			mcp.WithString(
				"body",
				mcp.Description("The body content of the issue."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the create issue operation */
func (tool *GithubCreateIssueTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubUpdateIssueTool implements a tool for updating an issue */
type GithubUpdateIssueTool struct {
	mcp.Tool
}

/* NewGithubUpdateIssueTool creates a new tool for updating issues */
func NewGithubUpdateIssueTool() *GithubUpdateIssueTool {
	return &GithubUpdateIssueTool{
		Tool: mcp.NewTool(
			"update_issue",
			mcp.WithDescription("A tool for updating an issue in a GitHub repository."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository containing the issue."),
				mcp.Required(),
			),
			mcp.WithString(
				"issue_number",
				mcp.Description("The number of the issue to update."),
				mcp.Required(),
			),
			mcp.WithString(
				"title",
				mcp.Description("The new title of the issue."),
				mcp.Required(),
			),
			mcp.WithString(
				"body",
				mcp.Description("The new body content of the issue."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the update issue operation */
func (tool *GithubUpdateIssueTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubCloseIssueTool implements a tool for closing an issue */
type GithubCloseIssueTool struct {
	mcp.Tool
}

/* NewGithubCloseIssueTool creates a new tool for closing issues */
func NewGithubCloseIssueTool() *GithubCloseIssueTool {
	return &GithubCloseIssueTool{
		Tool: mcp.NewTool(
			"close_issue",
			mcp.WithDescription("A tool for closing an issue in a GitHub repository."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository containing the issue."),
				mcp.Required(),
			),
			mcp.WithString(
				"issue_number",
				mcp.Description("The number of the issue to close."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the close issue operation */
func (tool *GithubCloseIssueTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubListPullsTool implements a tool for listing pull requests */
type GithubListPullsTool struct {
	mcp.Tool
}

/* NewGithubListPullsTool creates a new tool for listing pull requests */
func NewGithubListPullsTool() *GithubListPullsTool {
	return &GithubListPullsTool{
		Tool: mcp.NewTool(
			"list_pulls",
			mcp.WithDescription("A tool for listing pull requests in a GitHub repository."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository to list pull requests from."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the list pulls operation */
func (tool *GithubListPullsTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubCreatePullTool implements a tool for creating a pull request */
type GithubCreatePullTool struct {
	mcp.Tool
}

/* NewGithubCreatePullTool creates a new tool for creating pull requests */
func NewGithubCreatePullTool() *GithubCreatePullTool {
	return &GithubCreatePullTool{
		Tool: mcp.NewTool(
			"create_pull",
			mcp.WithDescription("A tool for creating a pull request in a GitHub repository."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository to create the pull request in."),
				mcp.Required(),
			),
			mcp.WithString(
				"title",
				mcp.Description("The title of the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"head",
				mcp.Description("The name of the branch where your changes are implemented."),
				mcp.Required(),
			),
			mcp.WithString(
				"base",
				mcp.Description("The name of the branch you want your changes pulled into."),
				mcp.Required(),
			),
			mcp.WithString(
				"body",
				mcp.Description("The contents of the pull request."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the create pull request operation */
func (tool *GithubCreatePullTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubUpdatePullTool implements a tool for updating a pull request */
type GithubUpdatePullTool struct {
	mcp.Tool
}

/* NewGithubUpdatePullTool creates a new tool for updating pull requests */
func NewGithubUpdatePullTool() *GithubUpdatePullTool {
	return &GithubUpdatePullTool{
		Tool: mcp.NewTool(
			"update_pull",
			mcp.WithDescription("A tool for updating a pull request in a GitHub repository."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository containing the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"pull_number",
				mcp.Description("The number of the pull request to update."),
				mcp.Required(),
			),
			mcp.WithString(
				"title",
				mcp.Description("The new title of the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"body",
				mcp.Description("The new contents of the pull request."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the update pull request operation */
func (tool *GithubUpdatePullTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubMergePullTool implements a tool for merging a pull request */
type GithubMergePullTool struct {
	mcp.Tool
}

/* NewGithubMergePullTool creates a new tool for merging pull requests */
func NewGithubMergePullTool() *GithubMergePullTool {
	return &GithubMergePullTool{
		Tool: mcp.NewTool(
			"merge_pull",
			mcp.WithDescription("A tool for merging a pull request in a GitHub repository."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository containing the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"pull_number",
				mcp.Description("The number of the pull request to merge."),
				mcp.Required(),
			),
			mcp.WithString(
				"merge_method",
				mcp.Description("The merge method to use (merge, squash, or rebase)."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the merge pull request operation */
func (tool *GithubMergePullTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubListReviewsTool implements a tool for listing pull request reviews */
type GithubListReviewsTool struct {
	mcp.Tool
}

/* NewGithubListReviewsTool creates a new tool for listing reviews */
func NewGithubListReviewsTool() *GithubListReviewsTool {
	return &GithubListReviewsTool{
		Tool: mcp.NewTool(
			"list_reviews",
			mcp.WithDescription("A tool for listing reviews on a GitHub pull request."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository containing the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"pull_number",
				mcp.Description("The number of the pull request."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the list reviews operation */
func (tool *GithubListReviewsTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubCreateReviewTool implements a tool for creating a pull request review */
type GithubCreateReviewTool struct {
	mcp.Tool
}

/* NewGithubCreateReviewTool creates a new tool for creating reviews */
func NewGithubCreateReviewTool() *GithubCreateReviewTool {
	return &GithubCreateReviewTool{
		Tool: mcp.NewTool(
			"create_review",
			mcp.WithDescription("A tool for creating a review on a GitHub pull request."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository containing the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"pull_number",
				mcp.Description("The number of the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"event",
				mcp.Description("The review action (APPROVE, REQUEST_CHANGES, or COMMENT)."),
				mcp.Required(),
			),
			mcp.WithString(
				"body",
				mcp.Description("The review comment."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the create review operation */
func (tool *GithubCreateReviewTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubUpdateReviewTool implements a tool for updating a pull request review */
type GithubUpdateReviewTool struct {
	mcp.Tool
}

/* NewGithubUpdateReviewTool creates a new tool for updating reviews */
func NewGithubUpdateReviewTool() *GithubUpdateReviewTool {
	return &GithubUpdateReviewTool{
		Tool: mcp.NewTool(
			"update_review",
			mcp.WithDescription("A tool for updating a review on a GitHub pull request."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository containing the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"pull_number",
				mcp.Description("The number of the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"review_id",
				mcp.Description("The ID of the review to update."),
				mcp.Required(),
			),
			mcp.WithString(
				"body",
				mcp.Description("The updated review comment."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the update review operation */
func (tool *GithubUpdateReviewTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubListReviewCommentsTool implements a tool for listing review comments */
type GithubListReviewCommentsTool struct {
	mcp.Tool
}

/* NewGithubListReviewCommentsTool creates a new tool for listing review comments */
func NewGithubListReviewCommentsTool() *GithubListReviewCommentsTool {
	return &GithubListReviewCommentsTool{
		Tool: mcp.NewTool(
			"list_review_comments",
			mcp.WithDescription("A tool for listing review comments on a GitHub pull request."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository containing the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"pull_number",
				mcp.Description("The number of the pull request."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the list review comments operation */
func (tool *GithubListReviewCommentsTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* GithubCreateReviewCommentTool implements a tool for creating a review comment */
type GithubCreateReviewCommentTool struct {
	mcp.Tool
}

/* NewGithubCreateReviewCommentTool creates a new tool for creating review comments */
func NewGithubCreateReviewCommentTool() *GithubCreateReviewCommentTool {
	return &GithubCreateReviewCommentTool{
		Tool: mcp.NewTool(
			"create_review_comment",
			mcp.WithDescription("A tool for creating a review comment on a GitHub pull request."),
			mcp.WithString(
				"repository",
				mcp.Description("The repository containing the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"pull_number",
				mcp.Description("The number of the pull request."),
				mcp.Required(),
			),
			mcp.WithString(
				"commit_id",
				mcp.Description("The SHA of the commit to comment on."),
				mcp.Required(),
			),
			mcp.WithString(
				"path",
				mcp.Description("The relative path to the file being commented on."),
				mcp.Required(),
			),
			mcp.WithString(
				"position",
				mcp.Description("The line index in the diff to comment on."),
				mcp.Required(),
			),
			mcp.WithString(
				"body",
				mcp.Description("The text of the review comment."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the create review comment operation */
func (tool *GithubCreateReviewCommentTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}
