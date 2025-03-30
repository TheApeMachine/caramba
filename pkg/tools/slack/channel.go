package slack

import (
	"github.com/slack-go/slack"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
GetChannelInfo retrieves information about a specific Slack channel.

Takes a channel ID and returns detailed information about the channel.
Returns an error if the retrieval fails.
*/
func (client *Client) GetChannelInfo(channel string) (interface{}, error) {
	return client.conn.GetConversationInfo(&slack.GetConversationInfoInput{
		ChannelID: channel,
	})
}

/*
ListChannels retrieves a list of all accessible Slack channels.

Returns a map containing the list of channels and a cursor for pagination.
Returns an error if the retrieval fails.
*/
func (client *Client) ListChannels() (interface{}, error) {
	channels, nextCursor, err := client.conn.GetConversations(&slack.GetConversationsParameters{})
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{
		"channels": channels,
		"cursor":   nextCursor,
	}, nil
}

/*
CreateChannel creates a new public channel in Slack.

Takes a channel name and creates a new public conversation.
Returns an error if the creation fails.
*/
func (client *Client) CreateChannel(channel string) (interface{}, error) {
	return client.conn.CreateConversation(slack.CreateConversationParams{
		ChannelName: channel,
		IsPrivate:   false,
	})
}

/*
GetThreadReplies retrieves all replies in a message thread.

Takes a channel ID and thread timestamp. Returns a map containing the messages,
pagination information, and whether there are more messages to retrieve.
Returns an error if the thread timestamp is missing or if retrieval fails.
*/
func (client *Client) GetThreadReplies(channel, threadTS string) (interface{}, error) {
	if threadTS == "" {
		return nil, errnie.Error("thread_ts is required for get_thread_replies operation")
	}

	messages, hasMore, nextCursor, err := client.conn.GetConversationReplies(&slack.GetConversationRepliesParameters{
		ChannelID: channel,
		Timestamp: threadTS,
	})
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{
		"messages": messages,
		"hasMore":  hasMore,
		"cursor":   nextCursor,
	}, nil
}
