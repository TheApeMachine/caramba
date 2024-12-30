package provider

import "strings"

/*
Thread is a collection of messages, which will be used to generate a response.
This is a generic type to keep track of the conversation history, and will be
converted to the appropriate format by each provider.
*/
type Thread struct {
	Messages []*Message
}

/*
NewThread creates a new thread.
*/
func NewThread() *Thread {
	return &Thread{
		Messages: make([]*Message, 0),
	}
}

func (thread *Thread) String() string {
	builder := strings.Builder{}

	for _, message := range thread.Messages {
		builder.WriteString(message.Content + "\n")
	}

	return builder.String()
}

/*
AddMessage adds a message to the thread.
*/
func (thread *Thread) AddMessage(message *Message) {
	thread.Messages = append(thread.Messages, message)
}

func (thread *Thread) Scrathpad() *Message {
	// Remove the most message from the thread.
	message := thread.Messages[:len(thread.Messages)-1]

	return message
}