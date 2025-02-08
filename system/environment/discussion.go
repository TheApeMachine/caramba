package environment

import (
	"github.com/theapemachine/caramba/stream"
)

/*
Discussion is an environment where multiple agents are orchestrated to have
a multi-turn conversation, examining a topic, strategy, or other debatable
concept. It is meant to align agents, and explore alternative ideas.
*/
type Discussion struct {
	owner        stream.Generator
	participants map[string]stream.Generator
}

func NewDiscussion(
	owner stream.Generator, participants map[string]stream.Generator,
) *Discussion {
	return &Discussion{
		owner:        owner,
		participants: participants,
	}
}