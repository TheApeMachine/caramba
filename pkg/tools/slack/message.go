package slack

import (
	"encoding/json"

	"github.com/slack-go/slack"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
PostMessage sends a new message to a Slack channel.

Takes a channel ID, message text, optional thread timestamp, and an artifact containing
additional message options like blocks and attachments. Returns the channel ID and
message timestamp on success, or an error if the post fails.
*/
func (client *Client) PostMessage(channel, text, threadTS string, artifact map[string]interface{}) (interface{}, error) {
	msgOptions := []slack.MsgOption{
		slack.MsgOptionText(text, false),
	}

	if threadTS != "" {
		msgOptions = append(msgOptions, slack.MsgOptionTS(threadTS))
	}

	if blocks := artifact["blocks"].(string); blocks != "" {
		var msgBlocks []slack.Block
		if err := json.Unmarshal([]byte(blocks), &msgBlocks); err == nil {
			msgOptions = append(msgOptions, slack.MsgOptionBlocks(msgBlocks...))
		}
	}

	if attachments := artifact["attachments"].(string); attachments != "" {
		var msgAttachments []slack.Attachment
		if err := json.Unmarshal([]byte(attachments), &msgAttachments); err == nil {
			msgOptions = append(msgOptions, slack.MsgOptionAttachments(msgAttachments...))
		}
	}

	channelID, timestamp, err := client.conn.PostMessage(channel, msgOptions...)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"channel": channelID,
		"ts":      timestamp,
	}, nil
}

/*
UpdateMessage updates an existing message in a Slack channel.

Takes a channel ID, message timestamp, and new text.
Returns an error if the timestamp is missing or if the update fails.
*/
func (client *Client) UpdateMessage(channel, threadTS, text string) error {
	if threadTS == "" {
		return errnie.Error("thread_ts is required for update_message operation")
	}

	_, _, _, err := client.conn.UpdateMessage(channel, threadTS, slack.MsgOptionText(text, false))
	return err
}

/*
DeleteMessage removes a message from a Slack channel.

Takes a channel ID and message timestamp.
Returns an error if the timestamp is missing or if the deletion fails.
*/
func (client *Client) DeleteMessage(channel, threadTS string) error {
	if threadTS == "" {
		return errnie.Error("thread_ts is required for delete_message operation")
	}

	_, _, err := client.conn.DeleteMessage(channel, threadTS)
	return err
}

/*
SearchMessages searches for messages across all accessible Slack channels.

Takes a search text string and returns matching messages.
Returns an error if the search text is empty or if the search fails.
*/
func (client *Client) SearchMessages(text string) (interface{}, error) {
	if text == "" {
		return nil, errnie.Error("text is required for search_messages operation")
	}

	return client.conn.SearchMessages(text, slack.SearchParameters{})
}
