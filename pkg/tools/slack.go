package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools/slack"
)

/*
SlackTool provides a streaming interface to Slack operations.
It manages Slack API interactions through a buffered client connection
and implements io.ReadWriteCloser for streaming data processing.
*/
type SlackTool struct {
	*ToolBuilder
	pctx   context.Context
	ctx    context.Context
	cancel context.CancelFunc
	client *slack.Client
}

type SlackToolOption func(*SlackTool)

/*
NewSlackTool creates a new Slack tool instance.

It initializes a Slack client and sets up a buffered stream for
processing Slack operations. The buffer copies data bidirectionally
between the artifact and the Slack client.
*/
func NewSlackTool(opts ...SlackToolOption) *SlackTool {
	ctx, cancel := context.WithCancel(context.Background())

	client := slack.NewClient()

	tool := &SlackTool{
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

func WithSlackCancel(ctx context.Context) SlackToolOption {
	return func(tool *SlackTool) {
		tool.pctx = ctx
	}
}

func (tool *SlackTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("slack.SlackTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		for {
			select {
			case <-tool.pctx.Done():
				errnie.Debug("slack.SlackTool.Generate: parent context done")
				tool.cancel()
				return
			case <-tool.ctx.Done():
				errnie.Debug("slack.SlackTool.Generate: context done")
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

// SlackPostMessageTool implements a tool for posting messages to Slack
type SlackPostMessageTool struct {
	*SlackTool
}

func NewSlackPostMessageTool() *SlackPostMessageTool {
	// Create MCP tool definition based on schema from config.yml
	postMessageTool := mcp.NewTool(
		"post_message",
		mcp.WithDescription("A tool for posting messages to Slack workspaces and channels."),
		mcp.WithString(
			"channel",
			mcp.Description("The channel to post the message to."),
			mcp.Required(),
		),
		mcp.WithString(
			"message",
			mcp.Description("The message to post to the channel."),
			mcp.Required(),
		),
	)

	spmt := &SlackPostMessageTool{
		SlackTool: NewSlackTool(),
	}

	spmt.ToolBuilder.mcp = &postMessageTool
	return spmt
}

func (tool *SlackPostMessageTool) ID() string {
	return "slack_post_message"
}

func (tool *SlackPostMessageTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackPostMessageTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for posting messages
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackPostMessageTool
func (tool *SlackPostMessageTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// SlackUploadFileTool implements a tool for uploading files to Slack
type SlackUploadFileTool struct {
	*SlackTool
}

func NewSlackUploadFileTool() *SlackUploadFileTool {
	// Create MCP tool definition based on schema from config.yml
	uploadFileTool := mcp.NewTool(
		"upload_file",
		mcp.WithDescription("A tool for uploading files to Slack workspaces and channels."),
		mcp.WithString(
			"channel",
			mcp.Description("The channel to upload the file to."),
			mcp.Required(),
		),
		mcp.WithString(
			"file",
			mcp.Description("The file to upload to the channel."),
			mcp.Required(),
		),
	)

	suft := &SlackUploadFileTool{
		SlackTool: NewSlackTool(),
	}

	suft.ToolBuilder.mcp = &uploadFileTool
	return suft
}

func (tool *SlackUploadFileTool) ID() string {
	return "slack_upload_file"
}

func (tool *SlackUploadFileTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackUploadFileTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for uploading files
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackUploadFileTool
func (tool *SlackUploadFileTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// SlackAddReactionTool implements a tool for adding reactions to messages
type SlackAddReactionTool struct {
	*SlackTool
}

func NewSlackAddReactionTool() *SlackAddReactionTool {
	// Create MCP tool definition based on schema from config.yml
	addReactionTool := mcp.NewTool(
		"add_reaction",
		mcp.WithDescription("A tool for adding reactions to messages in Slack workspaces and channels."),
		mcp.WithString(
			"channel",
			mcp.Description("The channel to add the reaction to."),
			mcp.Required(),
		),
		mcp.WithString(
			"reaction",
			mcp.Description("The reaction to add to the message."),
			mcp.Required(),
		),
	)

	sart := &SlackAddReactionTool{
		SlackTool: NewSlackTool(),
	}

	sart.ToolBuilder.mcp = &addReactionTool
	return sart
}

func (tool *SlackAddReactionTool) ID() string {
	return "slack_add_reaction"
}

func (tool *SlackAddReactionTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackAddReactionTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for adding reactions
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackAddReactionTool
func (tool *SlackAddReactionTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// SlackRemoveReactionTool implements a tool for removing reactions from messages
type SlackRemoveReactionTool struct {
	*SlackTool
}

func NewSlackRemoveReactionTool() *SlackRemoveReactionTool {
	// Create MCP tool definition based on schema from config.yml
	removeReactionTool := mcp.NewTool(
		"remove_reaction",
		mcp.WithDescription("A tool for removing reactions from messages in Slack workspaces and channels."),
		mcp.WithString(
			"channel",
			mcp.Description("The channel to remove the reaction from."),
			mcp.Required(),
		),
		mcp.WithString(
			"reaction",
			mcp.Description("The reaction to remove from the message."),
			mcp.Required(),
		),
	)

	srrt := &SlackRemoveReactionTool{
		SlackTool: NewSlackTool(),
	}

	srrt.ToolBuilder.mcp = &removeReactionTool
	return srrt
}

func (tool *SlackRemoveReactionTool) ID() string {
	return "slack_remove_reaction"
}

func (tool *SlackRemoveReactionTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackRemoveReactionTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for removing reactions
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackRemoveReactionTool
func (tool *SlackRemoveReactionTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// SlackGetChannelInfoTool implements a tool for getting channel information
type SlackGetChannelInfoTool struct {
	*SlackTool
}

func NewSlackGetChannelInfoTool() *SlackGetChannelInfoTool {
	// Create MCP tool definition based on schema from config.yml
	getChannelInfoTool := mcp.NewTool(
		"get_channel_info",
		mcp.WithDescription("A tool for getting information about a Slack channel."),
		mcp.WithString(
			"channel",
			mcp.Description("The channel to get information about."),
			mcp.Required(),
		),
	)

	sgcit := &SlackGetChannelInfoTool{
		SlackTool: NewSlackTool(),
	}

	sgcit.ToolBuilder.mcp = &getChannelInfoTool
	return sgcit
}

func (tool *SlackGetChannelInfoTool) ID() string {
	return "slack_get_channel_info"
}

func (tool *SlackGetChannelInfoTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackGetChannelInfoTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for getting channel information
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackGetChannelInfoTool
func (tool *SlackGetChannelInfoTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// SlackListChannelsTool implements a tool for listing channels
type SlackListChannelsTool struct {
	*SlackTool
}

func NewSlackListChannelsTool() *SlackListChannelsTool {
	// Create MCP tool definition based on schema from config.yml
	listChannelsTool := mcp.NewTool(
		"list_channels",
		mcp.WithDescription("A tool for listing all Slack channels."),
	)

	slct := &SlackListChannelsTool{
		SlackTool: NewSlackTool(),
	}

	slct.ToolBuilder.mcp = &listChannelsTool
	return slct
}

func (tool *SlackListChannelsTool) ID() string {
	return "slack_list_channels"
}

func (tool *SlackListChannelsTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackListChannelsTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for listing channels
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackListChannelsTool
func (tool *SlackListChannelsTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// SlackCreateChannelTool implements a tool for creating a channel
type SlackCreateChannelTool struct {
	*SlackTool
}

func NewSlackCreateChannelTool() *SlackCreateChannelTool {
	// Create MCP tool definition based on schema from config.yml
	createChannelTool := mcp.NewTool(
		"create_channel",
		mcp.WithDescription("A tool for creating a new Slack channel."),
		mcp.WithString(
			"name",
			mcp.Description("The name of the channel to create."),
			mcp.Required(),
		),
	)

	scct := &SlackCreateChannelTool{
		SlackTool: NewSlackTool(),
	}

	scct.ToolBuilder.mcp = &createChannelTool
	return scct
}

func (tool *SlackCreateChannelTool) ID() string {
	return "slack_create_channel"
}

func (tool *SlackCreateChannelTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackCreateChannelTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for creating a channel
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackCreateChannelTool
func (tool *SlackCreateChannelTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// SlackGetThreadRepliesTool implements a tool for getting thread replies
type SlackGetThreadRepliesTool struct {
	*SlackTool
}

func NewSlackGetThreadRepliesTool() *SlackGetThreadRepliesTool {
	// Create MCP tool definition based on schema from config.yml
	getThreadRepliesTool := mcp.NewTool(
		"get_thread_replies",
		mcp.WithDescription("A tool for getting replies to a message in a Slack channel."),
		mcp.WithString(
			"channel",
			mcp.Description("The channel to get replies from."),
			mcp.Required(),
		),
		mcp.WithString(
			"thread_ts",
			mcp.Description("The timestamp of the message to get replies from."),
			mcp.Required(),
		),
	)

	sgtrt := &SlackGetThreadRepliesTool{
		SlackTool: NewSlackTool(),
	}

	sgtrt.ToolBuilder.mcp = &getThreadRepliesTool
	return sgtrt
}

func (tool *SlackGetThreadRepliesTool) ID() string {
	return "slack_get_thread_replies"
}

func (tool *SlackGetThreadRepliesTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackGetThreadRepliesTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for getting thread replies
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackGetThreadRepliesTool
func (tool *SlackGetThreadRepliesTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// SlackSearchMessagesTool implements a tool for searching messages
type SlackSearchMessagesTool struct {
	*SlackTool
}

func NewSlackSearchMessagesTool() *SlackSearchMessagesTool {
	// Create MCP tool definition based on schema from config.yml
	searchMessagesTool := mcp.NewTool(
		"search_messages",
		mcp.WithDescription("A tool for searching for messages in a Slack channel."),
		mcp.WithString(
			"channel",
			mcp.Description("The channel to search for messages in."),
			mcp.Required(),
		),
		mcp.WithString(
			"text",
			mcp.Description("The text content to search for."),
			mcp.Required(),
		),
		mcp.WithString(
			"thread_ts",
			mcp.Description("The timestamp of the message to search for."),
			mcp.Required(),
		),
	)

	ssmt := &SlackSearchMessagesTool{
		SlackTool: NewSlackTool(),
	}

	ssmt.ToolBuilder.mcp = &searchMessagesTool
	return ssmt
}

func (tool *SlackSearchMessagesTool) ID() string {
	return "slack_search_messages"
}

func (tool *SlackSearchMessagesTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackSearchMessagesTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for searching messages
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackSearchMessagesTool
func (tool *SlackSearchMessagesTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// SlackUpdateMessageTool implements a tool for updating messages
type SlackUpdateMessageTool struct {
	*SlackTool
}

func NewSlackUpdateMessageTool() *SlackUpdateMessageTool {
	// Create MCP tool definition based on schema from config.yml
	updateMessageTool := mcp.NewTool(
		"update_message",
		mcp.WithDescription("A tool for updating a message in a Slack channel."),
		mcp.WithString(
			"channel",
			mcp.Description("The channel to update the message in."),
			mcp.Required(),
		),
		mcp.WithString(
			"message_ts",
			mcp.Description("The timestamp of the message to update."),
			mcp.Required(),
		),
	)

	sumt := &SlackUpdateMessageTool{
		SlackTool: NewSlackTool(),
	}

	sumt.ToolBuilder.mcp = &updateMessageTool
	return sumt
}

func (tool *SlackUpdateMessageTool) ID() string {
	return "slack_update_message"
}

func (tool *SlackUpdateMessageTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackUpdateMessageTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for updating messages
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackUpdateMessageTool
func (tool *SlackUpdateMessageTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// SlackDeleteMessageTool implements a tool for deleting messages
type SlackDeleteMessageTool struct {
	*SlackTool
}

func NewSlackDeleteMessageTool() *SlackDeleteMessageTool {
	// Create MCP tool definition based on schema from config.yml
	deleteMessageTool := mcp.NewTool(
		"delete_message",
		mcp.WithDescription("A tool for deleting a message in a Slack channel."),
		mcp.WithString(
			"channel",
			mcp.Description("The channel to delete the message from."),
			mcp.Required(),
		),
		mcp.WithString(
			"message_ts",
			mcp.Description("The timestamp of the message to delete."),
			mcp.Required(),
		),
	)

	sdmt := &SlackDeleteMessageTool{
		SlackTool: NewSlackTool(),
	}

	sdmt.ToolBuilder.mcp = &deleteMessageTool
	return sdmt
}

func (tool *SlackDeleteMessageTool) ID() string {
	return "slack_delete_message"
}

func (tool *SlackDeleteMessageTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SlackTool.Generate(buffer, tool.fn)
}

func (tool *SlackDeleteMessageTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for deleting messages
	return artifact
}

// ToMCP returns the MCP tool definitions for the SlackDeleteMessageTool
func (tool *SlackDeleteMessageTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}
