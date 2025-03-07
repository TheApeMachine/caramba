package core

import (
	"bytes"
	"encoding/json"

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
	enc *json.Encoder
	dec *json.Decoder
	in  *bytes.Buffer
	out *bytes.Buffer
}

/*
NewMessage creates a new message with the given role, name, and content.
*/
func NewMessage(role string, name string, content string) *Message {
	errnie.Debug("NewMessage")

	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	msg := &Message{
		MessageData: &MessageData{
			Role:    role,
			Name:    name,
			Content: content,
		},
		enc: json.NewEncoder(out),
		dec: json.NewDecoder(in),
		in:  in,
		out: out,
	}

	// Pre-encode the message to JSON for reading
	msg.enc.Encode(msg.MessageData)

	return msg
}

/*
Read implements io.Reader for Message.

It reads from the internal buffer containing the JSON representation
*/
func (msg *Message) Read(p []byte) (n int, err error) {
	errnie.Debug("Message.Read")

	if msg.out.Len() == 0 {
		if err = errnie.NewErrIO(msg.enc.Encode(msg.MessageData)); err != nil {
			return 0, err
		}
	}

	return msg.out.Read(p)
}

/*
Write implements io.Writer for Message.

It updates the message content based on incoming data
*/
func (msg *Message) Write(p []byte) (n int, err error) {
	errnie.Debug("Message.Write", "p", string(p))

	// Reset the output buffer whenever we write new data
	if msg.out.Len() > 0 {
		msg.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = msg.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf MessageData
	if decErr := msg.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		msg.MessageData.Role = buf.Role
		msg.MessageData.Name = buf.Name
		msg.MessageData.Content = buf.Content

		// Re-encode to the output buffer for subsequent reads
		if encErr := msg.enc.Encode(msg.MessageData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
}

/*
Close implements io.Closer for Message
*/
func (msg *Message) Close() error {
	errnie.Debug("Message.Close")

	msg.MessageData.Role = ""
	msg.MessageData.Name = ""
	msg.MessageData.Content = ""

	msg.in.Reset()
	msg.out.Reset()

	return nil
}
