package core

import (
	"bytes"
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
MessageData holds the structured data for a message.
*/
type MessageData struct {
	Role    string `json:"role"`
	Name    string `json:"name,omitempty"`
	Content string `json:"content"`
}

/*
Message represents a message in the system and implements io.ReadWriteCloser.
*/
type Message struct {
	*MessageData
	*stream.Buffer
}

/*
NewMessage creates a new message with the provided role, name, and content.
*/
func NewMessage(role string, name string, content string) *Message {
	errnie.Debug("NewMessage", "role", role, "name", name, "content", content)

	msg := &Message{
		MessageData: &MessageData{
			Role:    role,
			Name:    name,
			Content: content,
		},
	}

	msg.Buffer = stream.NewBuffer(
		msg,
		msg,
		func(message any) error {
			msg.MessageData = message.(*Message).MessageData
			return nil
		},
	)

	// Pre-buffer the MessageData, since we have the values.
	buf := bytes.NewBuffer([]byte{})
	json.NewEncoder(buf).Encode(msg.MessageData)

	if _, err := io.Copy(msg, buf); err != nil {
		errnie.NewErrIO(err)
		return nil
	}

	return msg
}

/*
Read implements io.Reader for Message.
*/
func (msg *Message) Read(p []byte) (n int, err error) {
	errnie.Debug("Message.Read")
	return msg.Buffer.Read(p)
}

/*
Write implements io.Writer for Message.
*/
func (msg *Message) Write(p []byte) (n int, err error) {
	errnie.Debug("Message.Write", "p", string(p))
	return msg.Buffer.Write(p)
}

/*
Close implements io.Closer for Message
*/
func (msg *Message) Close() error {
	errnie.Debug("Message.Close")
	return msg.Buffer.Close()
}
