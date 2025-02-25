package tools

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/slack-go/slack"
)

// SlackTool provides integration with the Slack API
type SlackTool struct {
	// client is the Slack API client
	client *slack.Client
	// token is the Slack API token
	token string
	// defaultChannel is the default channel to use if none is specified
	defaultChannel string
}

// NewSlackTool creates a new SlackTool
func NewSlackTool(token string, defaultChannel string) *SlackTool {
	// Create Slack client
	client := slack.New(token)

	return &SlackTool{
		client:         client,
		token:          token,
		defaultChannel: defaultChannel,
	}
}

// Name returns the name of the tool
func (s *SlackTool) Name() string {
	return "slack"
}

// Description returns the description of the tool
func (s *SlackTool) Description() string {
	return "Integrates with Slack API for messaging, notifications, searching, and more"
}

// Execute executes the tool with the given arguments
func (s *SlackTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	action, ok := args["action"].(string)
	if !ok {
		return nil, errors.New("action must be a string")
	}

	switch action {
	case "post_message":
		return s.postMessage(ctx, args)
	case "get_channel_history":
		return s.getChannelHistory(ctx, args)
	case "search_messages":
		return s.searchMessages(ctx, args)
	case "add_reaction":
		return s.addReaction(ctx, args)
	case "get_thread":
		return s.getThread(ctx, args)
	case "list_channels":
		return s.listChannels(ctx, args)
	case "get_user_info":
		return s.getUserInfo(ctx, args)
	case "update_message":
		return s.updateMessage(ctx, args)
	case "delete_message":
		return s.deleteMessage(ctx, args)
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

// Schema returns the JSON schema for the tool's arguments
func (s *SlackTool) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"action": map[string]interface{}{
				"type": "string",
				"enum": []string{
					"post_message",
					"get_channel_history",
					"search_messages",
					"add_reaction",
					"get_thread",
					"list_channels",
					"get_user_info",
					"update_message",
					"delete_message",
				},
				"description": "Action to perform with the Slack API",
			},
			"channel": map[string]interface{}{
				"type":        "string",
				"description": "Channel ID or name to interact with",
			},
			"text": map[string]interface{}{
				"type":        "string",
				"description": "Text content for the message",
			},
			"thread_ts": map[string]interface{}{
				"type":        "string",
				"description": "Thread timestamp to reply to or get",
			},
			"reaction": map[string]interface{}{
				"type":        "string",
				"description": "Emoji name (without colons) to add as a reaction",
			},
			"timestamp": map[string]interface{}{
				"type":        "string",
				"description": "Message timestamp to react to or update",
			},
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Search query for messages",
			},
			"user_id": map[string]interface{}{
				"type":        "string",
				"description": "User ID to get information about",
			},
			"blocks": map[string]interface{}{
				"type":        "object",
				"description": "Block Kit blocks for rich message formatting",
			},
			"limit": map[string]interface{}{
				"type":        "number",
				"description": "Maximum number of results to return",
			},
		},
		"required": []string{"action"},
	}
}

// getChannel returns the channel ID from args or the default channel
func (s *SlackTool) getChannel(args map[string]interface{}) (string, error) {
	channel, ok := args["channel"].(string)
	if !ok || channel == "" {
		if s.defaultChannel == "" {
			return "", errors.New("channel must be specified")
		}
		return s.defaultChannel, nil
	}
	return channel, nil
}

// postMessage posts a message to a Slack channel
func (s *SlackTool) postMessage(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the channel
	channel, err := s.getChannel(args)
	if err != nil {
		return nil, err
	}

	// Get the message text
	text, ok := args["text"].(string)
	if !ok || text == "" {
		return nil, errors.New("text must be specified")
	}

	// Optional thread timestamp
	threadTS, _ := args["thread_ts"].(string)

	// Optional Block Kit blocks
	var blocks slack.Blocks
	if _, ok := args["blocks"].(map[string]interface{}); ok {
		// This is a simplified implementation - in a real system,
		// you'd want to properly parse the Block Kit format
		// For now, we'll just use text blocks
		blocks = slack.Blocks{
			BlockSet: []slack.Block{
				slack.NewSectionBlock(
					slack.NewTextBlockObject(slack.MarkdownType, text, false, false),
					nil,
					nil,
				),
			},
		}
	}

	// Post the message
	msgOptions := []slack.MsgOption{
		slack.MsgOptionText(text, false),
	}

	// Add blocks if specified
	if len(blocks.BlockSet) > 0 {
		msgOptions = append(msgOptions, slack.MsgOptionBlocks(blocks.BlockSet...))
	}

	// Add thread_ts if specified
	if threadTS != "" {
		msgOptions = append(msgOptions, slack.MsgOptionTS(threadTS))
	}

	// Post the message
	result, timestamp, err := s.client.PostMessageContext(ctx, channel, msgOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to post message: %w", err)
	}

	return map[string]interface{}{
		"channel":   result,
		"timestamp": timestamp,
		"text":      text,
	}, nil
}

// getChannelHistory gets the message history from a channel
func (s *SlackTool) getChannelHistory(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the channel
	channel, err := s.getChannel(args)
	if err != nil {
		return nil, err
	}

	// Set up parameters
	params := slack.GetConversationHistoryParameters{
		ChannelID: channel,
		Limit:     100, // Default limit
	}

	// Optional limit
	if limit, ok := args["limit"].(float64); ok {
		params.Limit = int(limit)
	}

	// Optional oldest timestamp
	if oldest, ok := args["oldest"].(string); ok {
		params.Oldest = oldest
	}

	// Optional latest timestamp
	if latest, ok := args["latest"].(string); ok {
		params.Latest = latest
	}

	// Get the history
	history, err := s.client.GetConversationHistoryContext(ctx, &params)
	if err != nil {
		return nil, fmt.Errorf("failed to get channel history: %w", err)
	}

	// Format the messages
	messages := make([]map[string]interface{}, 0, len(history.Messages))
	for _, msg := range history.Messages {
		message := map[string]interface{}{
			"text":      msg.Text,
			"timestamp": msg.Timestamp,
			"user":      msg.User,
			"has_thread": msg.ThreadTimestamp != "" &&
				msg.ThreadTimestamp != msg.Timestamp,
			"reply_count": msg.ReplyCount,
		}
		messages = append(messages, message)
	}

	return map[string]interface{}{
		"channel":  channel,
		"messages": messages,
		"has_more": history.HasMore,
	}, nil
}

// searchMessages searches for messages in Slack
func (s *SlackTool) searchMessages(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the search query
	query, ok := args["query"].(string)
	if !ok || query == "" {
		return nil, errors.New("query must be specified")
	}

	// Set up parameters
	params := slack.SearchParameters{
		Sort:  "timestamp",
		Count: 20, // Default count
	}

	// Optional count
	if count, ok := args["limit"].(float64); ok {
		params.Count = int(count)
	}

	// Search
	result, err := s.client.SearchMessagesContext(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to search messages: %w", err)
	}

	// Format the matches
	matches := make([]map[string]interface{}, 0, len(result.Matches))
	for _, match := range result.Matches {
		message := map[string]interface{}{
			"text":      match.Text,
			"timestamp": match.Timestamp,
			"user":      match.Username,
			"channel":   match.Channel.Name,
			"permalink": match.Permalink,
		}
		matches = append(matches, message)
	}

	return map[string]interface{}{
		"query":      query,
		"total":      result.Total,
		"pagination": result.Pagination,
		"paging":     result.Paging,
		"matches":    matches,
	}, nil
}

// addReaction adds a reaction emoji to a message
func (s *SlackTool) addReaction(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the channel
	channel, err := s.getChannel(args)
	if err != nil {
		return nil, err
	}

	// Get the message timestamp
	timestamp, ok := args["timestamp"].(string)
	if !ok || timestamp == "" {
		return nil, errors.New("timestamp must be specified")
	}

	// Get the reaction emoji (without colons)
	reaction, ok := args["reaction"].(string)
	if !ok || reaction == "" {
		return nil, errors.New("reaction must be specified")
	}

	// Add the reaction
	err = s.client.AddReactionContext(ctx, reaction, slack.ItemRef{
		Channel:   channel,
		Timestamp: timestamp,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to add reaction: %w", err)
	}

	return map[string]interface{}{
		"success":   true,
		"channel":   channel,
		"timestamp": timestamp,
		"reaction":  reaction,
	}, nil
}

// getThread gets a message thread
func (s *SlackTool) getThread(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the channel
	channel, err := s.getChannel(args)
	if err != nil {
		return nil, err
	}

	// Get the thread timestamp
	threadTS, ok := args["thread_ts"].(string)
	if !ok || threadTS == "" {
		return nil, errors.New("thread_ts must be specified")
	}

	// Set up parameters
	params := slack.GetConversationRepliesParameters{
		ChannelID: channel,
		Timestamp: threadTS,
		Limit:     100, // Default limit
	}

	// Optional limit
	if limit, ok := args["limit"].(float64); ok {
		params.Limit = int(limit)
	}

	// Get the thread
	messages, _, _, err := s.client.GetConversationRepliesContext(ctx, &params)
	if err != nil {
		return nil, fmt.Errorf("failed to get thread: %w", err)
	}

	// Format the messages
	threadMessages := make([]map[string]interface{}, 0, len(messages))
	for _, msg := range messages {
		message := map[string]interface{}{
			"text":      msg.Text,
			"timestamp": msg.Timestamp,
			"user":      msg.User,
			"is_parent": msg.Timestamp == threadTS,
		}
		threadMessages = append(threadMessages, message)
	}

	return map[string]interface{}{
		"channel":   channel,
		"thread_ts": threadTS,
		"messages":  threadMessages,
		"count":     len(messages),
	}, nil
}

// listChannels lists available Slack channels
func (s *SlackTool) listChannels(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Set up parameters
	params := slack.GetConversationsParameters{
		Types: []string{"public_channel", "private_channel"},
		Limit: 200, // Default limit
	}

	// Optional limit
	if limit, ok := args["limit"].(float64); ok {
		params.Limit = int(limit)
	}

	// Optional exclude_archived
	if excludeArchived, ok := args["exclude_archived"].(bool); ok && excludeArchived {
		params.ExcludeArchived = true
	}

	// List channels
	channels, nextCursor, err := s.client.GetConversationsContext(ctx, &params)
	if err != nil {
		return nil, fmt.Errorf("failed to list channels: %w", err)
	}

	// Format the channels
	channelList := make([]map[string]interface{}, 0, len(channels))
	for _, channel := range channels {
		channelInfo := map[string]interface{}{
			"id":          channel.ID,
			"name":        channel.Name,
			"is_channel":  channel.IsChannel,
			"is_private":  channel.IsPrivate,
			"is_archived": channel.IsArchived,
			"created":     time.Unix(int64(channel.Created), 0).String(),
			"creator":     channel.Creator,
			"members":     channel.NumMembers,
		}
		channelList = append(channelList, channelInfo)
	}

	return map[string]interface{}{
		"channels":    channelList,
		"count":       len(channels),
		"next_cursor": nextCursor,
	}, nil
}

// getUserInfo gets information about a Slack user
func (s *SlackTool) getUserInfo(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the user ID
	userID, ok := args["user_id"].(string)
	if !ok || userID == "" {
		return nil, errors.New("user_id must be specified")
	}

	// Get user info
	user, err := s.client.GetUserInfoContext(ctx, userID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user info: %w", err)
	}

	return map[string]interface{}{
		"id":          user.ID,
		"name":        user.Name,
		"real_name":   user.RealName,
		"title":       user.Profile.Title,
		"email":       user.Profile.Email,
		"phone":       user.Profile.Phone,
		"status_text": user.Profile.StatusText,
		"is_admin":    user.IsAdmin,
		"is_bot":      user.IsBot,
		"tz":          user.TZ,
		"updated":     time.Unix(int64(user.Updated), 0).String(),
	}, nil
}

// updateMessage updates an existing message
func (s *SlackTool) updateMessage(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the channel
	channel, err := s.getChannel(args)
	if err != nil {
		return nil, err
	}

	// Get the message timestamp
	timestamp, ok := args["timestamp"].(string)
	if !ok || timestamp == "" {
		return nil, errors.New("timestamp must be specified")
	}

	// Get the new message text
	text, ok := args["text"].(string)
	if !ok || text == "" {
		return nil, errors.New("text must be specified")
	}

	// Optional Block Kit blocks
	var blocks slack.Blocks
	if _, ok := args["blocks"].(map[string]interface{}); ok {
		// This is a simplified implementation - in a real system,
		// you'd want to properly parse the Block Kit format
		blocks = slack.Blocks{
			BlockSet: []slack.Block{
				slack.NewSectionBlock(
					slack.NewTextBlockObject(slack.MarkdownType, text, false, false),
					nil,
					nil,
				),
			},
		}
	}

	// Update options
	updateOptions := []slack.MsgOption{
		slack.MsgOptionText(text, false),
	}

	// Add blocks if specified
	if len(blocks.BlockSet) > 0 {
		updateOptions = append(updateOptions, slack.MsgOptionBlocks(blocks.BlockSet...))
	}

	// Update the message
	_, timestamp, _, err = s.client.UpdateMessageContext(ctx, channel, timestamp, updateOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to update message: %w", err)
	}

	return map[string]interface{}{
		"success":   true,
		"channel":   channel,
		"timestamp": timestamp,
		"text":      text,
	}, nil
}

// deleteMessage deletes a message
func (s *SlackTool) deleteMessage(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Get the channel
	channel, err := s.getChannel(args)
	if err != nil {
		return nil, err
	}

	// Get the message timestamp
	timestamp, ok := args["timestamp"].(string)
	if !ok || timestamp == "" {
		return nil, errors.New("timestamp must be specified")
	}

	// Delete the message
	_, _, err = s.client.DeleteMessageContext(ctx, channel, timestamp)
	if err != nil {
		return nil, fmt.Errorf("failed to delete message: %w", err)
	}

	return map[string]interface{}{
		"success":   true,
		"channel":   channel,
		"timestamp": timestamp,
	}, nil
}
