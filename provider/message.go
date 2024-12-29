package provider

/*
Message is a generic message for a role, which will be converted to the
appropriate format by each provider.
*/
type Message struct {
	Role    Role
	Content string
}

/*
NewMessage creates a generic message for a role, which will be
converted to the appropriate format by each provider.
*/
func NewMessage(role Role, content string) *Message {
	return &Message{
		Role:    role,
		Content: content,
	}
}

func (message *Message) Append(content string) {
	message.Content += content
}
