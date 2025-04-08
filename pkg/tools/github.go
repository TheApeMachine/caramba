package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
)

/* GithubTool provides a base for all GitHub operations */
type GithubTool struct {
	operations map[string]ToolType
}

/* NewGithubTool creates a new GitHub tool with all operations */
func NewGithubTool(artifact *datura.Artifact) *GithubTool {
	getRepositories := NewGithubGetRepositoriesTool(artifact)
	getRepository := NewGithubGetRepositoryTool(artifact)
	createRepository := NewGithubCreateRepositoryTool(artifact)
	deleteRepository := NewGithubDeleteRepositoryTool(artifact)
	updateRepository := NewGithubUpdateRepositoryTool(artifact)
	listIssues := NewGithubListIssuesTool(artifact)
	createIssue := NewGithubCreateIssueTool(artifact)
	updateIssue := NewGithubUpdateIssueTool(artifact)
	closeIssue := NewGithubCloseIssueTool(artifact)
	listPulls := NewGithubListPullsTool(artifact)
	createPull := NewGithubCreatePullTool(artifact)
	updatePull := NewGithubUpdatePullTool(artifact)
	mergePull := NewGithubMergePullTool(artifact)
	listReviews := NewGithubListReviewsTool(artifact)
	createReview := NewGithubCreateReviewTool(artifact)
	updateReview := NewGithubUpdateReviewTool(artifact)
	listReviewComments := NewGithubListReviewCommentsTool(artifact)
	createReviewComment := NewGithubCreateReviewCommentTool(artifact)

	return &GithubTool{
		operations: map[string]ToolType{
			"get_repositories":      {getRepositories.Tool, getRepositories.Use, getRepositories.UseMCP},
			"get_repository":        {getRepository.Tool, getRepository.Use, getRepository.UseMCP},
			"create_repository":     {createRepository.Tool, createRepository.Use, createRepository.UseMCP},
			"delete_repository":     {deleteRepository.Tool, deleteRepository.Use, deleteRepository.UseMCP},
			"update_repository":     {updateRepository.Tool, updateRepository.Use, updateRepository.UseMCP},
			"list_issues":           {listIssues.Tool, listIssues.Use, listIssues.UseMCP},
			"create_issue":          {createIssue.Tool, createIssue.Use, createIssue.UseMCP},
			"update_issue":          {updateIssue.Tool, updateIssue.Use, updateIssue.UseMCP},
			"close_issue":           {closeIssue.Tool, closeIssue.Use, closeIssue.UseMCP},
			"list_pulls":            {listPulls.Tool, listPulls.Use, listPulls.UseMCP},
			"create_pull":           {createPull.Tool, createPull.Use, createPull.UseMCP},
			"update_pull":           {updatePull.Tool, updatePull.Use, updatePull.UseMCP},
			"merge_pull":            {mergePull.Tool, mergePull.Use, mergePull.UseMCP},
			"list_reviews":          {listReviews.Tool, listReviews.Use, listReviews.UseMCP},
			"create_review":         {createReview.Tool, createReview.Use, createReview.UseMCP},
			"update_review":         {updateReview.Tool, updateReview.Use, updateReview.UseMCP},
			"list_review_comments":  {listReviewComments.Tool, listReviewComments.Use, listReviewComments.UseMCP},
			"create_review_comment": {createReviewComment.Tool, createReviewComment.Use, createReviewComment.UseMCP},
		},
	}
}

func (tool *GithubTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	toolName := datura.GetMetaValue[string](artifact, "tool")
	return tool.operations[toolName].Use(ctx, artifact)
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
func NewGithubGetRepositoriesTool(artifact *datura.Artifact) *GithubGetRepositoriesTool {
	return &GithubGetRepositoriesTool{
		Tool: mcp.NewTool(
			"get_repositories",
			mcp.WithDescription("A tool for getting repositories from GitHub."),
		),
	}
}

/* Use executes the get repositories operation and returns the results */
func (tool *GithubGetRepositoriesTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubGetRepositoriesTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubGetRepositoryTool implements a tool for getting a repository from GitHub */
type GithubGetRepositoryTool struct {
	mcp.Tool
}

/* NewGithubGetRepositoryTool creates a new tool for getting a repository */
func NewGithubGetRepositoryTool(artifact *datura.Artifact) *GithubGetRepositoryTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubGetRepositoryTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubCreateRepositoryTool implements a tool for creating a repository */
type GithubCreateRepositoryTool struct {
	mcp.Tool
}

/* NewGithubCreateRepositoryTool creates a new tool for creating repositories */
func NewGithubCreateRepositoryTool(artifact *datura.Artifact) *GithubCreateRepositoryTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubCreateRepositoryTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubDeleteRepositoryTool implements a tool for deleting a repository */
type GithubDeleteRepositoryTool struct {
	mcp.Tool
}

/* NewGithubDeleteRepositoryTool creates a new tool for deleting repositories */
func NewGithubDeleteRepositoryTool(artifact *datura.Artifact) *GithubDeleteRepositoryTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubDeleteRepositoryTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubUpdateRepositoryTool implements a tool for updating a repository */
type GithubUpdateRepositoryTool struct {
	mcp.Tool
}

/* NewGithubUpdateRepositoryTool creates a new tool for updating repositories */
func NewGithubUpdateRepositoryTool(artifact *datura.Artifact) *GithubUpdateRepositoryTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubUpdateRepositoryTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubListIssuesTool implements a tool for listing repository issues */
type GithubListIssuesTool struct {
	mcp.Tool
}

/* NewGithubListIssuesTool creates a new tool for listing issues */
func NewGithubListIssuesTool(artifact *datura.Artifact) *GithubListIssuesTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubListIssuesTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubCreateIssueTool implements a tool for creating an issue */
type GithubCreateIssueTool struct {
	mcp.Tool
}

/* NewGithubCreateIssueTool creates a new tool for creating issues */
func NewGithubCreateIssueTool(artifact *datura.Artifact) *GithubCreateIssueTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubCreateIssueTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubUpdateIssueTool implements a tool for updating an issue */
type GithubUpdateIssueTool struct {
	mcp.Tool
}

/* NewGithubUpdateIssueTool creates a new tool for updating issues */
func NewGithubUpdateIssueTool(artifact *datura.Artifact) *GithubUpdateIssueTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubUpdateIssueTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubCloseIssueTool implements a tool for closing an issue */
type GithubCloseIssueTool struct {
	mcp.Tool
}

/* NewGithubCloseIssueTool creates a new tool for closing issues */
func NewGithubCloseIssueTool(artifact *datura.Artifact) *GithubCloseIssueTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubCloseIssueTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubListPullsTool implements a tool for listing pull requests */
type GithubListPullsTool struct {
	mcp.Tool
}

/* NewGithubListPullsTool creates a new tool for listing pull requests */
func NewGithubListPullsTool(artifact *datura.Artifact) *GithubListPullsTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubListPullsTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubCreatePullTool implements a tool for creating a pull request */
type GithubCreatePullTool struct {
	mcp.Tool
}

/* NewGithubCreatePullTool creates a new tool for creating pull requests */
func NewGithubCreatePullTool(artifact *datura.Artifact) *GithubCreatePullTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubCreatePullTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubUpdatePullTool implements a tool for updating a pull request */
type GithubUpdatePullTool struct {
	mcp.Tool
}

/* NewGithubUpdatePullTool creates a new tool for updating pull requests */
func NewGithubUpdatePullTool(artifact *datura.Artifact) *GithubUpdatePullTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubUpdatePullTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubMergePullTool implements a tool for merging a pull request */
type GithubMergePullTool struct {
	mcp.Tool
}

/* NewGithubMergePullTool creates a new tool for merging pull requests */
func NewGithubMergePullTool(artifact *datura.Artifact) *GithubMergePullTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubMergePullTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubListReviewsTool implements a tool for listing pull request reviews */
type GithubListReviewsTool struct {
	mcp.Tool
}

/* NewGithubListReviewsTool creates a new tool for listing reviews */
func NewGithubListReviewsTool(artifact *datura.Artifact) *GithubListReviewsTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubListReviewsTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubCreateReviewTool implements a tool for creating a pull request review */
type GithubCreateReviewTool struct {
	mcp.Tool
}

/* NewGithubCreateReviewTool creates a new tool for creating reviews */
func NewGithubCreateReviewTool(artifact *datura.Artifact) *GithubCreateReviewTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubCreateReviewTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubUpdateReviewTool implements a tool for updating a pull request review */
type GithubUpdateReviewTool struct {
	mcp.Tool
}

/* NewGithubUpdateReviewTool creates a new tool for updating reviews */
func NewGithubUpdateReviewTool(artifact *datura.Artifact) *GithubUpdateReviewTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubUpdateReviewTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubListReviewCommentsTool implements a tool for listing review comments */
type GithubListReviewCommentsTool struct {
	mcp.Tool
}

/* NewGithubListReviewCommentsTool creates a new tool for listing review comments */
func NewGithubListReviewCommentsTool(artifact *datura.Artifact) *GithubListReviewCommentsTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubListReviewCommentsTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* GithubCreateReviewCommentTool implements a tool for creating a review comment */
type GithubCreateReviewCommentTool struct {
	mcp.Tool
}

/* NewGithubCreateReviewCommentTool creates a new tool for creating review comments */
func NewGithubCreateReviewCommentTool(artifact *datura.Artifact) *GithubCreateReviewCommentTool {
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *GithubCreateReviewCommentTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}
