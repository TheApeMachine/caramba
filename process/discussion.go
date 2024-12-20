package process

import "github.com/theapemachine/amsh/utils"

/*
Discussion is a process where multiple AI agents discuss a topic and come to a
consensus, which will be the final response of the process.
*/
type Discussion struct {
	Topic       string `json:"topic" jsonschema:"title=Topic,description=The topic to discuss,required"`
	NextSpeaker string `json:"next_speaker" jsonschema:"title=Next Speaker,description=The name of the next speaker,required"`
}

/*
SystemPrompt returns the system prompt for the Discussion process.
*/
func (discussion *Discussion) SystemPrompt(key string) string {
	return utils.SystemPrompt(key, "discussion", utils.GenerateSchema[Discussion]())
}
