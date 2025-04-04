package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
)

/* SlackTool provides a base for all Slack operations */
type SlackTool struct {
	operations map[string]ToolType
}

/* NewSlackTool creates a new Slack tool with all operations */
func NewSlackTool() *SlackTool {
	postMessage := NewSlackPostMessageTool()
	uploadFile := NewSlackUploadFileTool()
	addReaction := NewSlackAddReactionTool()
	removeReaction := NewSlackRemoveReactionTool()
	getChannelInfo := NewSlackGetChannelInfoTool()
	listChannels := NewSlackListChannelsTool()
	createChannel := NewSlackCreateChannelTool()
	getThreadReplies := NewSlackGetThreadRepliesTool()
	searchMessages := NewSlackSearchMessagesTool()
	updateMessage := NewSlackUpdateMessageTool()
	deleteMessage := NewSlackDeleteMessageTool()

	return &SlackTool{
		operations: map[string]ToolType{
			"post_message":       {postMessage.Tool, postMessage.Use},
			"upload_file":        {uploadFile.Tool, uploadFile.Use},
			"add_reaction":       {addReaction.Tool, addReaction.Use},
			"remove_reaction":    {removeReaction.Tool, removeReaction.Use},
			"get_channel_info":   {getChannelInfo.Tool, getChannelInfo.Use},
			"list_channels":      {listChannels.Tool, listChannels.Use},
			"create_channel":     {createChannel.Tool, createChannel.Use},
			"get_thread_replies": {getThreadReplies.Tool, getThreadReplies.Use},
			"search_messages":    {searchMessages.Tool, searchMessages.Use},
			"update_message":     {updateMessage.Tool, updateMessage.Use},
			"delete_message":     {deleteMessage.Tool, deleteMessage.Use},
		},
	}
}

/* ToMCP returns all Slack tool definitions */
func (tool *SlackTool) ToMCP() []ToolType {
	tools := make([]ToolType, 0)

	for _, tool := range tool.operations {
		tools = append(tools, tool)
	}

	return tools
}

/* SlackPostMessageTool implements a tool for posting messages to Slack */
type SlackPostMessageTool struct {
	mcp.Tool
}

/* NewSlackPostMessageTool creates a new tool for posting messages */
func NewSlackPostMessageTool() *SlackPostMessageTool {
	return &SlackPostMessageTool{
		Tool: mcp.NewTool(
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
		),
	}
}

/* Use executes the post message operation and returns the results */
func (tool *SlackPostMessageTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SlackUploadFileTool implements a tool for uploading files to Slack */
type SlackUploadFileTool struct {
	mcp.Tool
}

/* NewSlackUploadFileTool creates a new tool for uploading files */
func NewSlackUploadFileTool() *SlackUploadFileTool {
	return &SlackUploadFileTool{
		Tool: mcp.NewTool(
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
		),
	}
}

/* Use executes the upload file operation and returns the results */
func (tool *SlackUploadFileTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SlackAddReactionTool implements a tool for adding reactions to messages */
type SlackAddReactionTool struct {
	mcp.Tool
}

/* NewSlackAddReactionTool creates a new tool for adding reactions */
func NewSlackAddReactionTool() *SlackAddReactionTool {
	return &SlackAddReactionTool{
		Tool: mcp.NewTool(
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
		),
	}
}

/* Use executes the add reaction operation and returns the results */
func (tool *SlackAddReactionTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SlackRemoveReactionTool implements a tool for removing reactions from messages */
type SlackRemoveReactionTool struct {
	mcp.Tool
}

/* NewSlackRemoveReactionTool creates a new tool for removing reactions */
func NewSlackRemoveReactionTool() *SlackRemoveReactionTool {
	return &SlackRemoveReactionTool{
		Tool: mcp.NewTool(
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
		),
	}
}

/* Use executes the remove reaction operation and returns the results */
func (tool *SlackRemoveReactionTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SlackGetChannelInfoTool implements a tool for getting channel information */
type SlackGetChannelInfoTool struct {
	mcp.Tool
}

/* NewSlackGetChannelInfoTool creates a new tool for getting channel info */
func NewSlackGetChannelInfoTool() *SlackGetChannelInfoTool {
	return &SlackGetChannelInfoTool{
		Tool: mcp.NewTool(
			"get_channel_info",
			mcp.WithDescription("A tool for getting information about a Slack channel."),
			mcp.WithString(
				"channel",
				mcp.Description("The channel to get information about."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the get channel info operation and returns the results */
func (tool *SlackGetChannelInfoTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SlackListChannelsTool implements a tool for listing channels */
type SlackListChannelsTool struct {
	mcp.Tool
}

/* NewSlackListChannelsTool creates a new tool for listing channels */
func NewSlackListChannelsTool() *SlackListChannelsTool {
	return &SlackListChannelsTool{
		Tool: mcp.NewTool(
			"list_channels",
			mcp.WithDescription("A tool for listing all Slack channels."),
		),
	}
}

/* Use executes the list channels operation and returns the results */
func (tool *SlackListChannelsTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SlackCreateChannelTool implements a tool for creating a channel */
type SlackCreateChannelTool struct {
	mcp.Tool
}

/* NewSlackCreateChannelTool creates a new tool for creating channels */
func NewSlackCreateChannelTool() *SlackCreateChannelTool {
	return &SlackCreateChannelTool{
		Tool: mcp.NewTool(
			"create_channel",
			mcp.WithDescription("A tool for creating a new Slack channel."),
			mcp.WithString(
				"name",
				mcp.Description("The name of the channel to create."),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the create channel operation and returns the results */
func (tool *SlackCreateChannelTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SlackGetThreadRepliesTool implements a tool for getting thread replies */
type SlackGetThreadRepliesTool struct {
	mcp.Tool
}

/* NewSlackGetThreadRepliesTool creates a new tool for getting thread replies */
func NewSlackGetThreadRepliesTool() *SlackGetThreadRepliesTool {
	return &SlackGetThreadRepliesTool{
		Tool: mcp.NewTool(
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
		),
	}
}

/* Use executes the get thread replies operation and returns the results */
func (tool *SlackGetThreadRepliesTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SlackSearchMessagesTool implements a tool for searching messages */
type SlackSearchMessagesTool struct {
	mcp.Tool
}

/* NewSlackSearchMessagesTool creates a new tool for searching messages */
func NewSlackSearchMessagesTool() *SlackSearchMessagesTool {
	return &SlackSearchMessagesTool{
		Tool: mcp.NewTool(
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
		),
	}
}

/* Use executes the search messages operation and returns the results */
func (tool *SlackSearchMessagesTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SlackUpdateMessageTool implements a tool for updating messages */
type SlackUpdateMessageTool struct {
	mcp.Tool
}

/* NewSlackUpdateMessageTool creates a new tool for updating messages */
func NewSlackUpdateMessageTool() *SlackUpdateMessageTool {
	return &SlackUpdateMessageTool{
		Tool: mcp.NewTool(
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
		),
	}
}

/* Use executes the update message operation and returns the results */
func (tool *SlackUpdateMessageTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SlackDeleteMessageTool implements a tool for deleting messages */
type SlackDeleteMessageTool struct {
	mcp.Tool
}

/* NewSlackDeleteMessageTool creates a new tool for deleting messages */
func NewSlackDeleteMessageTool() *SlackDeleteMessageTool {
	return &SlackDeleteMessageTool{
		Tool: mcp.NewTool(
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
		),
	}
}

/* Use executes the delete message operation and returns the results */
func (tool *SlackDeleteMessageTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}
