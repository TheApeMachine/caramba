package slack

import (
	"encoding/json"

	"github.com/slack-go/slack"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func (client *Client) PostMessage(channel, text, threadTS string, artifact *datura.Artifact) (interface{}, error) {
	msgOptions := []slack.MsgOption{
		slack.MsgOptionText(text, false),
	}

	if threadTS != "" {
		msgOptions = append(msgOptions, slack.MsgOptionTS(threadTS))
	}

	if blocks := datura.GetMetaValue[string](artifact, "blocks"); blocks != "" {
		var msgBlocks []slack.Block
		if err := json.Unmarshal([]byte(blocks), &msgBlocks); err == nil {
			msgOptions = append(msgOptions, slack.MsgOptionBlocks(msgBlocks...))
		}
	}

	if attachments := datura.GetMetaValue[string](artifact, "attachments"); attachments != "" {
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

func (client *Client) UpdateMessage(channel, threadTS, text string) error {
	if threadTS == "" {
		return errnie.Error("thread_ts is required for update_message operation")
	}

	_, _, _, err := client.conn.UpdateMessage(channel, threadTS, slack.MsgOptionText(text, false))
	return err
}

func (client *Client) DeleteMessage(channel, threadTS string) error {
	if threadTS == "" {
		return errnie.Error("thread_ts is required for delete_message operation")
	}

	_, _, err := client.conn.DeleteMessage(channel, threadTS)
	return err
}

func (client *Client) SearchMessages(text string) (interface{}, error) {
	if text == "" {
		return nil, errnie.Error("text is required for search_messages operation")
	}

	return client.conn.SearchMessages(text, slack.SearchParameters{})
}
