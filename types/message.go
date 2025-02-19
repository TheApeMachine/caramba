package types

import (
	"fmt"
	"strings"
)

type Message struct {
	From    string      `json:"from"`
	To      string      `json:"to"`
	Content string      `json:"content"`
	Data    interface{} `json:"data,omitempty"`
}

func NewMessage(from, to, content string, data interface{}) *Message {
	return &Message{
		From:    from,
		To:      to,
		Content: content,
		Data:    data,
	}
}

func (m *Message) String() string {
	msg := []string{
		"<message>",
		"\t<from>" + m.From + "</from>",
		"\t<to>" + m.To + "</to>",
		"\t<content>" + m.Content + "</content>",
		"\t<data>\n" + fmt.Sprintf("\t\t%v", m.Data) + "\n\t</data>",
		"</message>",
	}

	return strings.Join(msg, "\n")
}
