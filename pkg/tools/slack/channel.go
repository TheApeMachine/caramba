package slack

import (
	"github.com/slack-go/slack"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func (client *Client) GetChannelInfo(channel string) (interface{}, error) {
	return client.conn.GetConversationInfo(&slack.GetConversationInfoInput{
		ChannelID: channel,
	})
}

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

func (client *Client) CreateChannel(channel string) (interface{}, error) {
	return client.conn.CreateConversation(slack.CreateConversationParams{
		ChannelName: channel,
		IsPrivate:   false,
	})
}

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
