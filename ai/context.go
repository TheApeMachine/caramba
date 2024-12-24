package ai

import (
	"strings"

	"github.com/theapemachine/caramba/process/mechanic"
	"github.com/theapemachine/caramba/process/prompt"
	"github.com/theapemachine/caramba/process/review"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/caramba/utils"
)

/*
Context is a wrapper around the user prompt, the current context, and feedback.
This allows us to pass around the context, and update it as we go, while the
reviewers can store their feedback in a map, which uses the agent name as key,
so we can inject it into the agent's context, once the agent has another turn.
*/
type Context struct {
	userPrompt      string
	currentContext  strings.Builder
	cleanedContext  strings.Builder
	feedback        map[string]string
	vectorQuery     *Agent
	graphQuery      *Agent
	currentAgent    *Agent
	vectorMemorizer *Agent
	graphMemorizer  *Agent
	reviewer        *Agent
	promptEngineer  *Agent
	mechanic        *Agent
}

/*
NewContext creates a new context, with the given user prompt.
*/
func NewContext(userPrompt string) *Context {
	return &Context{
		userPrompt:      utils.QuickWrap("user-prompt", userPrompt),
		feedback:        make(map[string]string),
		vectorQuery:     NewAgent("vector-query", tools.NewQdrantQuery("caramba", 1536), 2),
		graphQuery:      NewAgent("graph-query", tools.NewNeo4jQuery(), 2),
		vectorMemorizer: NewAgent("vector-memorizer", tools.NewQdrantStore("caramba", 1536), 2),
		graphMemorizer:  NewAgent("graph-memorizer", tools.NewNeo4jStore(), 2),
		reviewer:        NewAgent("reviewer", &review.Process{}, 1),
		promptEngineer:  NewAgent("prompt-engineer", &prompt.Process{}, 1),
		mechanic:        NewAgent("mechanic", &mechanic.Process{}, 1),
	}
}

/*
Feedback returns any pending feedback for the given agent name.
*/
func (ctx *Context) Feedback(name string) string {
	return utils.QuickWrap("feedback", ctx.feedback[name])
}

/*
SetCurrentAgent sets the current agent, so we can use it to generate feedback.
*/
func (ctx *Context) SetCurrentAgent(agent *Agent) {
	ctx.currentAgent = agent
}

/*
Generate produces the next context, wrapping the agent's response into a process
that provides a pre- and post-flight setup. High-level, this comes down to the
following:

- Query, and potentially inject, memories from the vector store.
- Query, and potentially inject, memories from the graph store.
- Generate the agent's response.
- Review the agent's response.
- Dynamically adjust/cleanup the current context.
- Extract, and potentially store, memories for the vector store.
- Extract, and potentially store, memories for the graph store.
- Evaluate the overall performance, and potentially adjust the agent's internals.
*/
func (ctx *Context) Generate() <-chan provider.Event {
	out := make(chan provider.Event)

	go func() {
		defer close(out)

		for _, agent := range []*Agent{
			ctx.vectorQuery,
			ctx.graphQuery,
			ctx.currentAgent,
			ctx.reviewer,
			ctx.promptEngineer,
			ctx.vectorMemorizer,
			ctx.graphMemorizer,
			ctx.mechanic,
		} {
			agent.buffer.Reset()

			agent.buffer.Poke(provider.Message{
				Role:    "user",
				Content: ctx.userPrompt,
			})

			agent.buffer.Poke(provider.Message{
				Role:    "assistant",
				Content: ctx.currentContext.String(),
			})

			for event := range agent.Generate() {
				out <- event
			}

			ctx.currentContext.WriteString(agent.buffer.String())
		}
	}()

	return out
}

/*
extractBlocks extracts any markdown code blocks from the current response.
*/
func (ctx *Context) extractBlocks(response string) {
	blocks := utils.ExtractJSONBlocks(response)

	for _, blockMap := range blocks {
		if _, ok := blockMap["improved"].(string); !ok {
			// Reset the current context, and replace with the cleaned context.
			ctx.currentContext.Reset()
			ctx.currentContext.WriteString(blockMap["improved"].(string))
		}
	}
}
