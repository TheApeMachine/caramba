package comms

import "github.com/theapemachine/caramba/utils"

type Message struct {
	To      string `json:"to" jsonschema:"title=To,description=The recipient of the message,required"`
	From    string `json:"from"`
	Subject string `json:"subject" jsonschema:"title=Subject,description=The subject of the message,required"`
	Body    string `json:"body" jsonschema:"title=Body,description=The body of the message,required"`
}

func (message *Message) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Message]()
}
