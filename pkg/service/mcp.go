package service

import (
	"context"
	"net/http"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/memory"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools"
)

// Define artifact roles as string constants
const (
	ArtifactRoleMemoryTool      = "memory_tool"
	ArtifactRoleAgentTool       = "agent_tool"
	ArtifactRoleEditorTool      = "editor_tool"
	ArtifactRoleGithubTool      = "github_tool"
	ArtifactRoleAzureTool       = "azure_tool"
	ArtifactRoleTrengoTool      = "trengo_tool"
	ArtifactRoleBrowserTool     = "browser_tool"
	ArtifactRoleEnvironmentTool = "environment_tool"
	ArtifactRoleSlackTool       = "slack_tool"
	ArtifactRoleSystemTool      = "system_tool"
)

type MCP struct {
	stdio *server.MCPServer
	sse   *server.SSEServer
	tools map[string]stream.Generator
}

func NewMCP() *MCP {
	errnie.Debug("NewMCP")

	// Initialize memory tools
	memoryTool := tools.NewMemoryTool(
		tools.WithStores(memory.NewQdrant(), memory.NewNeo4j()),
	)

	// Initialize specific sub-tools
	editorReadTool := tools.NewEditorReadTool()
	githubGetReposTool := tools.NewGithubGetRepositoriesTool()
	azureCreateWorkItemTool := tools.NewAzureCreateWorkItemTool()
	trengoListTicketsTool := tools.NewTrengoListTicketsTool()
	browserGetContentTool := tools.NewBrowserGetContentTool()
	environmentCommandTool := tools.NewEnvironmentCommandTool()
	agentBuilder := ai.NewAgentBuilder()
	slackSearchMessagesTool := tools.NewSlackSearchMessagesTool()
	systemInspectTool := tools.NewSystemInspectTool()

	return &MCP{
		stdio: server.NewMCPServer(
			"caramba-server",
			"1.0.0",
			server.WithResourceCapabilities(true, true),
			server.WithPromptCapabilities(true),
			server.WithToolCapabilities(true),
		),
		sse: server.NewSSEServer(
			server.NewMCPServer(
				"caramba-server",
				"1.0.0",
				server.WithResourceCapabilities(true, true),
				server.WithPromptCapabilities(true),
				server.WithToolCapabilities(true),
			),
			server.WithBaseURL("http://localhost:8080"),
			server.WithSSEContextFunc(authFromRequest),
		),
		tools: map[string]stream.Generator{
			"memory":      memoryTool,
			"editor":      editorReadTool,
			"github":      githubGetReposTool,
			"azure":       azureCreateWorkItemTool,
			"trengo":      trengoListTicketsTool,
			"browser":     browserGetContentTool,
			"environment": environmentCommandTool,
			"agent":       agentBuilder,
			"slack":       slackSearchMessagesTool,
			"system":      systemInspectTool,
		},
	}
}

func (service *MCP) Start() error {
	errnie.Debug("MCP.Start")

	service.stdio.AddTool(
		mcp.NewTool(
			"memory",
			mcp.WithDescription("A tool for working with memory stores."),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.memory.tool", req)
			return service.runToolGenerator(service.tools["memory"], &req, ArtifactRoleMemoryTool)
		},
	)

	service.stdio.AddTool(
		mcp.NewTool(
			"agent",
			mcp.WithDescription("A tool for working with agents."),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.agent.tool", req)
			return service.runToolGenerator(service.tools["agent"], &req, ArtifactRoleAgentTool)
		},
	)

	// Editor tools
	service.stdio.AddTool(
		service.tools["editor"].(*tools.EditorReadTool).ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.editor.tool", req)
			return service.runToolGenerator(service.tools["editor"], &req, ArtifactRoleEditorTool)
		},
	)

	editorWriteTool := tools.NewEditorWriteTool()
	service.stdio.AddTool(
		editorWriteTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.editor.write.tool", req)
			return service.runToolGenerator(editorWriteTool, &req, ArtifactRoleEditorTool)
		},
	)

	editorDeleteTool := tools.NewEditorDeleteTool()
	service.stdio.AddTool(
		editorDeleteTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.editor.delete.tool", req)
			return service.runToolGenerator(editorDeleteTool, &req, ArtifactRoleEditorTool)
		},
	)

	editorReplaceLinesTool := tools.NewEditorReplaceLinesTool()
	service.stdio.AddTool(
		editorReplaceLinesTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.editor.replace_lines.tool", req)
			return service.runToolGenerator(editorReplaceLinesTool, &req, ArtifactRoleEditorTool)
		},
	)

	editorInsertLinesTool := tools.NewEditorInsertLinesTool()
	service.stdio.AddTool(
		editorInsertLinesTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.editor.insert_lines.tool", req)
			return service.runToolGenerator(editorInsertLinesTool, &req, ArtifactRoleEditorTool)
		},
	)

	editorDeleteLinesTool := tools.NewEditorDeleteLinesTool()
	service.stdio.AddTool(
		editorDeleteLinesTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.editor.delete_lines.tool", req)
			return service.runToolGenerator(editorDeleteLinesTool, &req, ArtifactRoleEditorTool)
		},
	)

	editorReadLinesTool := tools.NewEditorReadLinesTool()
	service.stdio.AddTool(
		editorReadLinesTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.editor.read_lines.tool", req)
			return service.runToolGenerator(editorReadLinesTool, &req, ArtifactRoleEditorTool)
		},
	)

	// GitHub tools
	service.stdio.AddTool(
		service.tools["github"].(*tools.GithubGetRepositoriesTool).ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.tool", req)
			return service.runToolGenerator(service.tools["github"], &req, ArtifactRoleGithubTool)
		},
	)

	// Add more GitHub tools
	githubGetRepoTool := tools.NewGithubGetRepositoryTool()
	service.stdio.AddTool(
		githubGetRepoTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.get_repository.tool", req)
			return service.runToolGenerator(githubGetRepoTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubCreateRepoTool := tools.NewGithubCreateRepositoryTool()
	service.stdio.AddTool(
		githubCreateRepoTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.create_repository.tool", req)
			return service.runToolGenerator(githubCreateRepoTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubListBranchesTool := tools.NewGithubListBranchesTool()
	service.stdio.AddTool(
		githubListBranchesTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.list_branches.tool", req)
			return service.runToolGenerator(githubListBranchesTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubGetContentsTool := tools.NewGithubGetContentsTool()
	service.stdio.AddTool(
		githubGetContentsTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.get_contents.tool", req)
			return service.runToolGenerator(githubGetContentsTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubListPRsTool := tools.NewGithubListPullRequestsTool()
	service.stdio.AddTool(
		githubListPRsTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.list_pull_requests.tool", req)
			return service.runToolGenerator(githubListPRsTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubGetPRTool := tools.NewGithubGetPullRequestTool()
	service.stdio.AddTool(
		githubGetPRTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.get_pull_request.tool", req)
			return service.runToolGenerator(githubGetPRTool, &req, ArtifactRoleGithubTool)
		},
	)

	// Add more GitHub tools for PRs, issues, comments, and reviews
	githubCreatePRTool := tools.NewGithubCreatePullRequestTool()
	service.stdio.AddTool(
		githubCreatePRTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.create_pull_request.tool", req)
			return service.runToolGenerator(githubCreatePRTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubUpdatePRTool := tools.NewGithubUpdatePullRequestTool()
	service.stdio.AddTool(
		githubUpdatePRTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.update_pull_request.tool", req)
			return service.runToolGenerator(githubUpdatePRTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubListIssuesTool := tools.NewGithubListIssuesTool()
	service.stdio.AddTool(
		githubListIssuesTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.list_issues.tool", req)
			return service.runToolGenerator(githubListIssuesTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubGetIssueTool := tools.NewGithubGetIssueTool()
	service.stdio.AddTool(
		githubGetIssueTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.get_issue.tool", req)
			return service.runToolGenerator(githubGetIssueTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubCreateIssueTool := tools.NewGithubCreateIssueTool()
	service.stdio.AddTool(
		githubCreateIssueTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.create_issue.tool", req)
			return service.runToolGenerator(githubCreateIssueTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubUpdateIssueTool := tools.NewGithubUpdateIssueTool()
	service.stdio.AddTool(
		githubUpdateIssueTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.update_issue.tool", req)
			return service.runToolGenerator(githubUpdateIssueTool, &req, ArtifactRoleGithubTool)
		},
	)

	// Add GitHub PR comments and reviews tools
	githubCreatePRCommentTool := tools.NewGithubCreatePRCommentTool()
	service.stdio.AddTool(
		githubCreatePRCommentTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.create_pr_comment.tool", req)
			return service.runToolGenerator(githubCreatePRCommentTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubListPRCommentsTool := tools.NewGithubListPRCommentsTool()
	service.stdio.AddTool(
		githubListPRCommentsTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.list_pr_comments.tool", req)
			return service.runToolGenerator(githubListPRCommentsTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubCreatePRReviewTool := tools.NewGithubCreatePRReviewTool()
	service.stdio.AddTool(
		githubCreatePRReviewTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.create_pr_review.tool", req)
			return service.runToolGenerator(githubCreatePRReviewTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubListPRReviewsTool := tools.NewGithubListPRReviewsTool()
	service.stdio.AddTool(
		githubListPRReviewsTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.list_pr_reviews.tool", req)
			return service.runToolGenerator(githubListPRReviewsTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubCreateReviewCommentTool := tools.NewGithubCreateReviewCommentTool()
	service.stdio.AddTool(
		githubCreateReviewCommentTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.create_review_comment.tool", req)
			return service.runToolGenerator(githubCreateReviewCommentTool, &req, ArtifactRoleGithubTool)
		},
	)

	githubListReviewCommentsTool := tools.NewGithubListReviewCommentsTool()
	service.stdio.AddTool(
		githubListReviewCommentsTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.list_review_comments.tool", req)
			return service.runToolGenerator(githubListReviewCommentsTool, &req, ArtifactRoleGithubTool)
		},
	)

	// Azure tools
	service.stdio.AddTool(
		service.tools["azure"].(*tools.AzureCreateWorkItemTool).ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.azure.tool", req)
			return service.runToolGenerator(service.tools["azure"], &req, ArtifactRoleAzureTool)
		},
	)

	// Add more Azure tools
	azureUpdateWorkItemTool := tools.NewAzureUpdateWorkItemTool()
	service.stdio.AddTool(
		azureUpdateWorkItemTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.azure.update_work_item.tool", req)
			return service.runToolGenerator(azureUpdateWorkItemTool, &req, ArtifactRoleAzureTool)
		},
	)

	azureGetWorkItemTool := tools.NewAzureGetWorkItemTool()
	service.stdio.AddTool(
		azureGetWorkItemTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.azure.get_work_item.tool", req)
			return service.runToolGenerator(azureGetWorkItemTool, &req, ArtifactRoleAzureTool)
		},
	)

	azureListWorkItemsTool := tools.NewAzureListWorkItemsTool()
	service.stdio.AddTool(
		azureListWorkItemsTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.azure.list_work_items.tool", req)
			return service.runToolGenerator(azureListWorkItemsTool, &req, ArtifactRoleAzureTool)
		},
	)

	// Add Azure Wiki tools
	azureCreateWikiPageTool := tools.NewAzureCreateWikiPageTool()
	service.stdio.AddTool(
		azureCreateWikiPageTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.azure.create_wiki_page.tool", req)
			return service.runToolGenerator(azureCreateWikiPageTool, &req, ArtifactRoleAzureTool)
		},
	)

	azureUpdateWikiPageTool := tools.NewAzureUpdateWikiPageTool()
	service.stdio.AddTool(
		azureUpdateWikiPageTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.azure.update_wiki_page.tool", req)
			return service.runToolGenerator(azureUpdateWikiPageTool, &req, ArtifactRoleAzureTool)
		},
	)

	azureGetWikiPageTool := tools.NewAzureGetWikiPageTool()
	service.stdio.AddTool(
		azureGetWikiPageTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.azure.get_wiki_page.tool", req)
			return service.runToolGenerator(azureGetWikiPageTool, &req, ArtifactRoleAzureTool)
		},
	)

	azureListWikiPagesTool := tools.NewAzureListWikiPagesTool()
	service.stdio.AddTool(
		azureListWikiPagesTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.azure.list_wiki_pages.tool", req)
			return service.runToolGenerator(azureListWikiPagesTool, &req, ArtifactRoleAzureTool)
		},
	)

	// Trengo tools
	service.stdio.AddTool(
		service.tools["trengo"].(*tools.TrengoListTicketsTool).ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.tool", req)
			return service.runToolGenerator(service.tools["trengo"], &req, ArtifactRoleTrengoTool)
		},
	)

	// Add more Trengo tools
	trengoCreateTicketTool := tools.NewTrengoCreateTicketTool()
	service.stdio.AddTool(
		trengoCreateTicketTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.create_ticket.tool", req)
			return service.runToolGenerator(trengoCreateTicketTool, &req, ArtifactRoleTrengoTool)
		},
	)

	trengoAssignTicketTool := tools.NewTrengoAssignTicketTool()
	service.stdio.AddTool(
		trengoAssignTicketTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.assign_ticket.tool", req)
			return service.runToolGenerator(trengoAssignTicketTool, &req, ArtifactRoleTrengoTool)
		},
	)

	trengoCloseTicketTool := tools.NewTrengoCloseTicketTool()
	service.stdio.AddTool(
		trengoCloseTicketTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.close_ticket.tool", req)
			return service.runToolGenerator(trengoCloseTicketTool, &req, ArtifactRoleTrengoTool)
		},
	)

	// Add more Trengo tools
	trengoReopenTicketTool := tools.NewTrengoReopenTicketTool()
	service.stdio.AddTool(
		trengoReopenTicketTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.reopen_ticket.tool", req)
			return service.runToolGenerator(trengoReopenTicketTool, &req, ArtifactRoleTrengoTool)
		},
	)

	trengoListLabelsTool := tools.NewTrengoListLabelsTool()
	service.stdio.AddTool(
		trengoListLabelsTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.list_labels.tool", req)
			return service.runToolGenerator(trengoListLabelsTool, &req, ArtifactRoleTrengoTool)
		},
	)

	trengoGetLabelTool := tools.NewTrengoGetLabelTool()
	service.stdio.AddTool(
		trengoGetLabelTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.get_label.tool", req)
			return service.runToolGenerator(trengoGetLabelTool, &req, ArtifactRoleTrengoTool)
		},
	)

	trengoCreateLabelTool := tools.NewTrengoCreateLabelTool()
	service.stdio.AddTool(
		trengoCreateLabelTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.create_label.tool", req)
			return service.runToolGenerator(trengoCreateLabelTool, &req, ArtifactRoleTrengoTool)
		},
	)

	trengoUpdateLabelTool := tools.NewTrengoUpdateLabelTool()
	service.stdio.AddTool(
		trengoUpdateLabelTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.update_label.tool", req)
			return service.runToolGenerator(trengoUpdateLabelTool, &req, ArtifactRoleTrengoTool)
		},
	)

	trengoDeleteLabelTool := tools.NewTrengoDeleteLabelTool()
	service.stdio.AddTool(
		trengoDeleteLabelTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.delete_label.tool", req)
			return service.runToolGenerator(trengoDeleteLabelTool, &req, ArtifactRoleTrengoTool)
		},
	)

	// Browser tools
	service.stdio.AddTool(
		service.tools["browser"].(*tools.BrowserGetContentTool).ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.browser.tool", req)
			return service.runToolGenerator(service.tools["browser"], &req, ArtifactRoleBrowserTool)
		},
	)

	// Add more Browser tools
	browserGetLinksTool := tools.NewBrowserGetLinksTool()
	service.stdio.AddTool(
		browserGetLinksTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.browser.get_links.tool", req)
			return service.runToolGenerator(browserGetLinksTool, &req, ArtifactRoleBrowserTool)
		},
	)

	// Environment tools
	service.stdio.AddTool(
		service.tools["environment"].(*tools.EnvironmentCommandTool).ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.environment.tool", req)
			return service.runToolGenerator(service.tools["environment"], &req, ArtifactRoleEnvironmentTool)
		},
	)

	// Add more Environment tools
	environmentInputTool := tools.NewEnvironmentInputTool()
	service.stdio.AddTool(
		environmentInputTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.environment.input.tool", req)
			return service.runToolGenerator(environmentInputTool, &req, ArtifactRoleEnvironmentTool)
		},
	)

	// Slack tools
	service.stdio.AddTool(
		service.tools["slack"].(*tools.SlackSearchMessagesTool).ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.tool", req)
			return service.runToolGenerator(service.tools["slack"], &req, ArtifactRoleSlackTool)
		},
	)

	// Add more Slack tools
	slackPostMessageTool := tools.NewSlackPostMessageTool()
	service.stdio.AddTool(
		slackPostMessageTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.post_message.tool", req)
			return service.runToolGenerator(slackPostMessageTool, &req, ArtifactRoleSlackTool)
		},
	)

	slackAddReactionTool := tools.NewSlackAddReactionTool()
	service.stdio.AddTool(
		slackAddReactionTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.add_reaction.tool", req)
			return service.runToolGenerator(slackAddReactionTool, &req, ArtifactRoleSlackTool)
		},
	)

	// Add more Slack tools
	slackUploadFileTool := tools.NewSlackUploadFileTool()
	service.stdio.AddTool(
		slackUploadFileTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.upload_file.tool", req)
			return service.runToolGenerator(slackUploadFileTool, &req, ArtifactRoleSlackTool)
		},
	)

	slackRemoveReactionTool := tools.NewSlackRemoveReactionTool()
	service.stdio.AddTool(
		slackRemoveReactionTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.remove_reaction.tool", req)
			return service.runToolGenerator(slackRemoveReactionTool, &req, ArtifactRoleSlackTool)
		},
	)

	slackGetChannelInfoTool := tools.NewSlackGetChannelInfoTool()
	service.stdio.AddTool(
		slackGetChannelInfoTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.get_channel_info.tool", req)
			return service.runToolGenerator(slackGetChannelInfoTool, &req, ArtifactRoleSlackTool)
		},
	)

	slackListChannelsTool := tools.NewSlackListChannelsTool()
	service.stdio.AddTool(
		slackListChannelsTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.list_channels.tool", req)
			return service.runToolGenerator(slackListChannelsTool, &req, ArtifactRoleSlackTool)
		},
	)

	slackCreateChannelTool := tools.NewSlackCreateChannelTool()
	service.stdio.AddTool(
		slackCreateChannelTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.create_channel.tool", req)
			return service.runToolGenerator(slackCreateChannelTool, &req, ArtifactRoleSlackTool)
		},
	)

	slackGetThreadRepliesTool := tools.NewSlackGetThreadRepliesTool()
	service.stdio.AddTool(
		slackGetThreadRepliesTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.get_thread_replies.tool", req)
			return service.runToolGenerator(slackGetThreadRepliesTool, &req, ArtifactRoleSlackTool)
		},
	)

	slackUpdateMessageTool := tools.NewSlackUpdateMessageTool()
	service.stdio.AddTool(
		slackUpdateMessageTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.update_message.tool", req)
			return service.runToolGenerator(slackUpdateMessageTool, &req, ArtifactRoleSlackTool)
		},
	)

	slackDeleteMessageTool := tools.NewSlackDeleteMessageTool()
	service.stdio.AddTool(
		slackDeleteMessageTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.slack.delete_message.tool", req)
			return service.runToolGenerator(slackDeleteMessageTool, &req, ArtifactRoleSlackTool)
		},
	)

	// System tools
	systemInspectToolDef := mcp.NewTool(
		"system_inspect",
		mcp.WithDescription("A tool for inspecting the system."),
		mcp.WithString(
			"scope",
			mcp.Description("The scope of the inspection."),
			mcp.Enum("agents", "topics"),
			mcp.Required(),
		),
	)

	service.stdio.AddTool(
		systemInspectToolDef,
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.system.tool", req)
			return service.runToolGenerator(service.tools["system"], &req, ArtifactRoleSystemTool)
		},
	)

	// Add more System tools
	systemOptimizeTool := tools.NewSystemOptimizeTool()
	service.stdio.AddTool(
		systemOptimizeTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.system.optimize.tool", req)
			return service.runToolGenerator(systemOptimizeTool, &req, ArtifactRoleSystemTool)
		},
	)

	systemMessageTool := tools.NewSystemMessageTool()
	service.stdio.AddTool(
		systemMessageTool.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.system.message.tool", req)
			return service.runToolGenerator(systemMessageTool, &req, ArtifactRoleSystemTool)
		},
	)

	return nil
}

func (service *MCP) runToolGenerator(tool stream.Generator, req *mcp.CallToolRequest, role string) (*mcp.CallToolResult, error) {
	errnie.Debug("MCP.runToolGenerator")

	options := []datura.ArtifactOption{}

	for key, val := range req.Params.Arguments {
		options = append(options, datura.WithMeta(key, val))
	}

	artifact := datura.New(options...)

	input := make(chan *datura.Artifact, 1)
	input <- artifact
	close(input)

	output := tool.Generate(input)

	result := <-output

	payload, err := result.DecryptPayload()
	if err != nil {
		return mcp.NewToolResultText(errnie.Error(err).Error()), nil
	}

	return mcp.NewToolResultText(string(payload)), nil
}

func (service *MCP) Stop() error {
	errnie.Debug("MCP.Stop")
	return nil
}

type authKey struct{}

func authFromRequest(ctx context.Context, r *http.Request) context.Context {
	return withAuthKey(ctx, r.Header.Get("Authorization"))
}

func withAuthKey(ctx context.Context, auth string) context.Context {
	return context.WithValue(ctx, authKey{}, auth)
}
