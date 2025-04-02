package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools/github"
)

/*
Github provides a streaming interface to GitHub operations.
It manages GitHub API interactions through a buffered client connection
and implements io.ReadWriteCloser for streaming data processing.
*/
type GithubTool struct {
	*ToolBuilder
	pctx   context.Context
	ctx    context.Context
	cancel context.CancelFunc
	client *github.Client
}

type GithubToolOption func(*GithubTool)

/*
NewGithub creates a new GitHub tool instance.

It initializes a GitHub client and sets up a buffered stream for
processing GitHub operations. The buffer copies data bidirectionally
between the artifact and the GitHub client.
*/
func NewGithubTool(opts ...GithubToolOption) *GithubTool {
	ctx, cancel := context.WithCancel(context.Background())

	client := github.NewClient()

	tool := &GithubTool{
		ToolBuilder: NewToolBuilder(),
		ctx:         ctx,
		cancel:      cancel,
		client:      client,
	}

	for _, opt := range opts {
		opt(tool)
	}

	return tool
}

func WithGithubCancel(ctx context.Context) GithubToolOption {
	return func(tool *GithubTool) {
		tool.pctx = ctx
	}
}

func (tool *GithubTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("github.GithubTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		for {
			select {
			case <-tool.pctx.Done():
				errnie.Debug("github.GithubTool.Generate: parent context done")
				tool.cancel()
				return
			case <-tool.ctx.Done():
				errnie.Debug("github.GithubTool.Generate: context done")
				return
			case artifact := <-buffer:
				for _, f := range fn {
					out <- f(artifact)
				}
			}
		}
	}()

	return out
}

// GithubGetRepositoriesTool implements a tool for getting repositories from GitHub
type GithubGetRepositoriesTool struct {
	*GithubTool
}

func NewGithubGetRepositoriesTool() *GithubGetRepositoriesTool {
	// Create MCP tool definition based on schema from config.yml
	getReposTool := mcp.NewTool(
		"get_repositories",
		mcp.WithDescription("A tool for getting repositories from GitHub."),
	)

	grrt := &GithubGetRepositoriesTool{
		GithubTool: NewGithubTool(),
	}

	grrt.ToolBuilder.mcp = &getReposTool
	return grrt
}

func (tool *GithubGetRepositoriesTool) ID() string {
	return "github_get_repositories"
}

func (tool *GithubGetRepositoriesTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubGetRepositoriesTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for getting repositories
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubGetRepositoriesTool
func (tool *GithubGetRepositoriesTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubGetRepositoryTool implements a tool for getting a repository from GitHub
type GithubGetRepositoryTool struct {
	*GithubTool
}

func NewGithubGetRepositoryTool() *GithubGetRepositoryTool {
	// Create MCP tool definition based on schema from config.yml
	getRepoTool := mcp.NewTool(
		"get_repository",
		mcp.WithDescription("A tool for getting a repository from GitHub."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to get."),
			mcp.Required(),
		),
	)

	grt := &GithubGetRepositoryTool{
		GithubTool: NewGithubTool(),
	}

	grt.ToolBuilder.mcp = &getRepoTool
	return grt
}

func (tool *GithubGetRepositoryTool) ID() string {
	return "github_get_repository"
}

func (tool *GithubGetRepositoryTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubGetRepositoryTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for getting a repository
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubGetRepositoryTool
func (tool *GithubGetRepositoryTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubCreateRepositoryTool implements a tool for creating a repository
type GithubCreateRepositoryTool struct {
	*GithubTool
}

func NewGithubCreateRepositoryTool() *GithubCreateRepositoryTool {
	// Create MCP tool definition based on schema from config.yml
	createRepoTool := mcp.NewTool(
		"create_repository",
		mcp.WithDescription("A tool for creating a repository on GitHub."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to create."),
			mcp.Required(),
		),
	)

	gcrt := &GithubCreateRepositoryTool{
		GithubTool: NewGithubTool(),
	}

	gcrt.ToolBuilder.mcp = &createRepoTool
	return gcrt
}

func (tool *GithubCreateRepositoryTool) ID() string {
	return "github_create_repository"
}

func (tool *GithubCreateRepositoryTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubCreateRepositoryTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for creating a repository
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubCreateRepositoryTool
func (tool *GithubCreateRepositoryTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubListBranchesTool implements a tool for listing branches on a repository
type GithubListBranchesTool struct {
	*GithubTool
}

func NewGithubListBranchesTool() *GithubListBranchesTool {
	// Create MCP tool definition based on schema from config.yml
	listBranchesTool := mcp.NewTool(
		"list_branches",
		mcp.WithDescription("A tool for listing branches on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to list branches on."),
			mcp.Required(),
		),
	)

	glbt := &GithubListBranchesTool{
		GithubTool: NewGithubTool(),
	}

	glbt.ToolBuilder.mcp = &listBranchesTool
	return glbt
}

func (tool *GithubListBranchesTool) ID() string {
	return "github_list_branches"
}

func (tool *GithubListBranchesTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubListBranchesTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for listing branches on a repository
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubListBranchesTool
func (tool *GithubListBranchesTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubGetContentsTool implements a tool for getting the contents of a repository
type GithubGetContentsTool struct {
	*GithubTool
}

func NewGithubGetContentsTool() *GithubGetContentsTool {
	// Create MCP tool definition based on schema from config.yml
	getContentsTool := mcp.NewTool(
		"get_contents",
		mcp.WithDescription("A tool for getting the contents of a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to get the contents of."),
			mcp.Required(),
		),
	)

	ggct := &GithubGetContentsTool{
		GithubTool: NewGithubTool(),
	}

	ggct.ToolBuilder.mcp = &getContentsTool
	return ggct
}

func (tool *GithubGetContentsTool) ID() string {
	return "github_get_contents"
}

func (tool *GithubGetContentsTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubGetContentsTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for getting contents of a repository
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubGetContentsTool
func (tool *GithubGetContentsTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubListPullRequestsTool implements a tool for listing pull requests on a repository
type GithubListPullRequestsTool struct {
	*GithubTool
}

func NewGithubListPullRequestsTool() *GithubListPullRequestsTool {
	// Create MCP tool definition based on schema from config.yml
	listPRsTool := mcp.NewTool(
		"list_pull_requests",
		mcp.WithDescription("A tool for listing pull requests on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to list pull requests on."),
			mcp.Required(),
		),
	)

	glprt := &GithubListPullRequestsTool{
		GithubTool: NewGithubTool(),
	}

	glprt.ToolBuilder.mcp = &listPRsTool
	return glprt
}

func (tool *GithubListPullRequestsTool) ID() string {
	return "github_list_pull_requests"
}

func (tool *GithubListPullRequestsTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubListPullRequestsTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for listing pull requests
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubListPullRequestsTool
func (tool *GithubListPullRequestsTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubGetPullRequestTool implements a tool for getting a pull request
type GithubGetPullRequestTool struct {
	*GithubTool
}

func NewGithubGetPullRequestTool() *GithubGetPullRequestTool {
	// Create MCP tool definition based on schema from config.yml
	getPRTool := mcp.NewTool(
		"get_pull_request",
		mcp.WithDescription("A tool for getting a pull request on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to get the pull request from."),
			mcp.Required(),
		),
	)

	ggprt := &GithubGetPullRequestTool{
		GithubTool: NewGithubTool(),
	}

	ggprt.ToolBuilder.mcp = &getPRTool
	return ggprt
}

func (tool *GithubGetPullRequestTool) ID() string {
	return "github_get_pull_request"
}

func (tool *GithubGetPullRequestTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubGetPullRequestTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for getting a pull request
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubGetPullRequestTool
func (tool *GithubGetPullRequestTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubCreatePullRequestTool implements a tool for creating a pull request
type GithubCreatePullRequestTool struct {
	*GithubTool
}

func NewGithubCreatePullRequestTool() *GithubCreatePullRequestTool {
	// Create MCP tool definition based on schema from config.yml
	createPRTool := mcp.NewTool(
		"create_pull_request",
		mcp.WithDescription("A tool for creating a pull request on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to create the pull request on."),
			mcp.Required(),
		),
	)

	gcprt := &GithubCreatePullRequestTool{
		GithubTool: NewGithubTool(),
	}

	gcprt.ToolBuilder.mcp = &createPRTool
	return gcprt
}

func (tool *GithubCreatePullRequestTool) ID() string {
	return "github_create_pull_request"
}

func (tool *GithubCreatePullRequestTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubCreatePullRequestTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for creating a pull request
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubCreatePullRequestTool
func (tool *GithubCreatePullRequestTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubUpdatePullRequestTool implements a tool for updating a pull request
type GithubUpdatePullRequestTool struct {
	*GithubTool
}

func NewGithubUpdatePullRequestTool() *GithubUpdatePullRequestTool {
	// Create MCP tool definition based on schema from config.yml
	updatePRTool := mcp.NewTool(
		"update_pull_request",
		mcp.WithDescription("A tool for updating a pull request on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to update the pull request on."),
			mcp.Required(),
		),
	)

	guprt := &GithubUpdatePullRequestTool{
		GithubTool: NewGithubTool(),
	}

	guprt.ToolBuilder.mcp = &updatePRTool
	return guprt
}

func (tool *GithubUpdatePullRequestTool) ID() string {
	return "github_update_pull_request"
}

func (tool *GithubUpdatePullRequestTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubUpdatePullRequestTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for updating a pull request
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubUpdatePullRequestTool
func (tool *GithubUpdatePullRequestTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubListIssuesTool implements a tool for listing issues
type GithubListIssuesTool struct {
	*GithubTool
}

func NewGithubListIssuesTool() *GithubListIssuesTool {
	// Create MCP tool definition based on schema from config.yml
	listIssuesTool := mcp.NewTool(
		"list_issues",
		mcp.WithDescription("A tool for listing issues on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to list issues on."),
			mcp.Required(),
		),
	)

	glit := &GithubListIssuesTool{
		GithubTool: NewGithubTool(),
	}

	glit.ToolBuilder.mcp = &listIssuesTool
	return glit
}

func (tool *GithubListIssuesTool) ID() string {
	return "github_list_issues"
}

func (tool *GithubListIssuesTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubListIssuesTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for listing issues
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubListIssuesTool
func (tool *GithubListIssuesTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubGetIssueTool implements a tool for getting an issue
type GithubGetIssueTool struct {
	*GithubTool
}

func NewGithubGetIssueTool() *GithubGetIssueTool {
	// Create MCP tool definition based on schema from config.yml
	getIssueTool := mcp.NewTool(
		"get_issue",
		mcp.WithDescription("A tool for getting an issue on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to get the issue from."),
			mcp.Required(),
		),
	)

	ggit := &GithubGetIssueTool{
		GithubTool: NewGithubTool(),
	}

	ggit.ToolBuilder.mcp = &getIssueTool
	return ggit
}

func (tool *GithubGetIssueTool) ID() string {
	return "github_get_issue"
}

func (tool *GithubGetIssueTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubGetIssueTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for getting an issue
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubGetIssueTool
func (tool *GithubGetIssueTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubCreateIssueTool implements a tool for creating an issue
type GithubCreateIssueTool struct {
	*GithubTool
}

func NewGithubCreateIssueTool() *GithubCreateIssueTool {
	// Create MCP tool definition based on schema from config.yml
	createIssueTool := mcp.NewTool(
		"create_issue",
		mcp.WithDescription("A tool for creating an issue on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to create the issue on."),
			mcp.Required(),
		),
	)

	gcit := &GithubCreateIssueTool{
		GithubTool: NewGithubTool(),
	}

	gcit.ToolBuilder.mcp = &createIssueTool
	return gcit
}

func (tool *GithubCreateIssueTool) ID() string {
	return "github_create_issue"
}

func (tool *GithubCreateIssueTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubCreateIssueTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for creating an issue
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubCreateIssueTool
func (tool *GithubCreateIssueTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubUpdateIssueTool implements a tool for updating an issue
type GithubUpdateIssueTool struct {
	*GithubTool
}

func NewGithubUpdateIssueTool() *GithubUpdateIssueTool {
	// Create MCP tool definition based on schema from config.yml
	updateIssueTool := mcp.NewTool(
		"update_issue",
		mcp.WithDescription("A tool for updating an issue on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to update the issue on."),
			mcp.Required(),
		),
	)

	guit := &GithubUpdateIssueTool{
		GithubTool: NewGithubTool(),
	}

	guit.ToolBuilder.mcp = &updateIssueTool
	return guit
}

func (tool *GithubUpdateIssueTool) ID() string {
	return "github_update_issue"
}

func (tool *GithubUpdateIssueTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubUpdateIssueTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for updating an issue
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubUpdateIssueTool
func (tool *GithubUpdateIssueTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubCreatePRCommentTool implements a tool for creating a PR comment
type GithubCreatePRCommentTool struct {
	*GithubTool
}

func NewGithubCreatePRCommentTool() *GithubCreatePRCommentTool {
	// Create MCP tool definition based on schema from config.yml
	createPRCommentTool := mcp.NewTool(
		"create_pr_comment",
		mcp.WithDescription("A tool for creating a comment on a pull request on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to create the pull request comment on."),
			mcp.Required(),
		),
	)

	gcpct := &GithubCreatePRCommentTool{
		GithubTool: NewGithubTool(),
	}

	gcpct.ToolBuilder.mcp = &createPRCommentTool
	return gcpct
}

func (tool *GithubCreatePRCommentTool) ID() string {
	return "github_create_pr_comment"
}

func (tool *GithubCreatePRCommentTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubCreatePRCommentTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for creating a PR comment
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubCreatePRCommentTool
func (tool *GithubCreatePRCommentTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubListPRCommentsTool implements a tool for listing PR comments
type GithubListPRCommentsTool struct {
	*GithubTool
}

func NewGithubListPRCommentsTool() *GithubListPRCommentsTool {
	// Create MCP tool definition based on schema from config.yml
	listPRCommentsTool := mcp.NewTool(
		"list_pr_comments",
		mcp.WithDescription("A tool for listing comments on a pull request on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to list pull request comments on."),
			mcp.Required(),
		),
	)

	glpct := &GithubListPRCommentsTool{
		GithubTool: NewGithubTool(),
	}

	glpct.ToolBuilder.mcp = &listPRCommentsTool
	return glpct
}

func (tool *GithubListPRCommentsTool) ID() string {
	return "github_list_pr_comments"
}

func (tool *GithubListPRCommentsTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubListPRCommentsTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for listing PR comments
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubListPRCommentsTool
func (tool *GithubListPRCommentsTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubCreatePRReviewTool implements a tool for creating a PR review
type GithubCreatePRReviewTool struct {
	*GithubTool
}

func NewGithubCreatePRReviewTool() *GithubCreatePRReviewTool {
	// Create MCP tool definition based on schema from config.yml
	createPRReviewTool := mcp.NewTool(
		"create_pr_review",
		mcp.WithDescription("A tool for creating a review on a pull request on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to create the pull request review on."),
			mcp.Required(),
		),
	)

	gcprt := &GithubCreatePRReviewTool{
		GithubTool: NewGithubTool(),
	}

	gcprt.ToolBuilder.mcp = &createPRReviewTool
	return gcprt
}

func (tool *GithubCreatePRReviewTool) ID() string {
	return "github_create_pr_review"
}

func (tool *GithubCreatePRReviewTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubCreatePRReviewTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for creating a PR review
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubCreatePRReviewTool
func (tool *GithubCreatePRReviewTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubListPRReviewsTool implements a tool for listing PR reviews
type GithubListPRReviewsTool struct {
	*GithubTool
}

func NewGithubListPRReviewsTool() *GithubListPRReviewsTool {
	// Create MCP tool definition based on schema from config.yml
	listPRReviewsTool := mcp.NewTool(
		"list_pr_reviews",
		mcp.WithDescription("A tool for listing reviews on a pull request on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to list pull request reviews on."),
			mcp.Required(),
		),
	)

	glprt := &GithubListPRReviewsTool{
		GithubTool: NewGithubTool(),
	}

	glprt.ToolBuilder.mcp = &listPRReviewsTool
	return glprt
}

func (tool *GithubListPRReviewsTool) ID() string {
	return "github_list_pr_reviews"
}

func (tool *GithubListPRReviewsTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubListPRReviewsTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for listing PR reviews
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubListPRReviewsTool
func (tool *GithubListPRReviewsTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubCreateReviewCommentTool implements a tool for creating a review comment
type GithubCreateReviewCommentTool struct {
	*GithubTool
}

func NewGithubCreateReviewCommentTool() *GithubCreateReviewCommentTool {
	// Create MCP tool definition based on schema from config.yml
	createReviewCommentTool := mcp.NewTool(
		"create_review_comment",
		mcp.WithDescription("A tool for creating a review comment on a pull request on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to create the pull request review comment on."),
			mcp.Required(),
		),
	)

	gcrct := &GithubCreateReviewCommentTool{
		GithubTool: NewGithubTool(),
	}

	gcrct.ToolBuilder.mcp = &createReviewCommentTool
	return gcrct
}

func (tool *GithubCreateReviewCommentTool) ID() string {
	return "github_create_review_comment"
}

func (tool *GithubCreateReviewCommentTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubCreateReviewCommentTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for creating a review comment
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubCreateReviewCommentTool
func (tool *GithubCreateReviewCommentTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// GithubListReviewCommentsTool implements a tool for listing review comments
type GithubListReviewCommentsTool struct {
	*GithubTool
}

func NewGithubListReviewCommentsTool() *GithubListReviewCommentsTool {
	// Create MCP tool definition based on schema from config.yml
	listReviewCommentsTool := mcp.NewTool(
		"list_review_comments",
		mcp.WithDescription("A tool for listing review comments on a pull request on a repository."),
		mcp.WithString(
			"repository",
			mcp.Description("The repository to list review comments on."),
			mcp.Required(),
		),
	)

	glrct := &GithubListReviewCommentsTool{
		GithubTool: NewGithubTool(),
	}

	glrct.ToolBuilder.mcp = &listReviewCommentsTool
	return glrct
}

func (tool *GithubListReviewCommentsTool) ID() string {
	return "github_list_review_comments"
}

func (tool *GithubListReviewCommentsTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.GithubTool.Generate(buffer, tool.fn)
}

func (tool *GithubListReviewCommentsTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for listing review comments
	return artifact
}

// ToMCP returns the MCP tool definitions for the GithubListReviewCommentsTool
func (tool *GithubListReviewCommentsTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}
