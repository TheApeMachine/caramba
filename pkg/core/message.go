package core

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
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
	dec    *json.Decoder
	enc    *json.Encoder
	buffer *bufio.ReadWriter
}

/*
NewMessage creates a new message with the provided role, name, and content.
*/
func NewMessage(role string, name string, content string) *Message {
	errnie.Debug("NewMessage", "role", role, "name", name, "content", content)

	buf := bytes.NewBuffer([]byte{})

	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	msg := &Message{
		MessageData: &MessageData{
			Role:    role,
			Name:    name,
			Content: content,
		},
		buffer: buffer,
		dec:    json.NewDecoder(buffer),
		enc:    json.NewEncoder(buffer),
	}

	msg.enc.Encode(msg.MessageData)
	return msg
}

/*
Read implements io.Reader for Message.
*/
func (msg *Message) Read(p []byte) (n int, err error) {
	errnie.Debug("Message.Read")

	if err = msg.buffer.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if n, err = msg.buffer.Read(p); err != nil && err != io.EOF {
		errnie.NewErrIO(err)
	}

	errnie.Debug("Message.Read", "n", n, "err", err)

	return n, err
}

/*
Write implements io.Writer for Message.
*/
func (msg *Message) Write(p []byte) (n int, err error) {
	errnie.Debug("Message.Write", "p", string(p))

	if n, err = msg.buffer.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = msg.dec.Decode(msg.MessageData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = msg.enc.Encode(msg.MessageData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("Message.Write", "n", n, "err", err)

	return n, err
}

/*
Close implements io.Closer for Message
*/
func (msg *Message) Close() error {
	errnie.Debug("Message.Close")

	msg.buffer.Flush()
	msg.MessageData = nil
	msg.dec = nil
	msg.enc = nil

	return nil
}
