package ai

import (
	"strings"

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
	vectorMemorizer *Agent
	graphMemorizer  *Agent
	reviewer        *Agent
	promptEngineer  *Agent
}

/*
NewContext creates a new context, with the given user prompt.
*/
func NewContext(userPrompt string) *Context {
	return &Context{
		userPrompt:      utils.QuickWrap("user-prompt", userPrompt),
		feedback:        make(map[string]string),
		vectorQuery:     NewAgent("vector-query", tools.NewQdrantQuery("caramba", 1536), 1),
		graphQuery:      NewAgent("graph-query", tools.NewNeo4jQuery(), 1),
		vectorMemorizer: NewAgent("vector-memorizer", tools.NewQdrantStore("caramba", 1536), 1),
		graphMemorizer:  NewAgent("graph-memorizer", tools.NewNeo4jStore(), 1),
		reviewer:        NewAgent("reviewer", &review.Process{}, 1),
		promptEngineer:  NewAgent("prompt-engineer", &prompt.Process{}, 1),
	}
}

/*
Feedback returns any pending feedback for the given agent name.
*/
func (ctx *Context) Feedback(name string) string {
	return utils.QuickWrap("feedback", ctx.feedback[name])
}

/*
Peek returns the cleaned context, which is the context after all the feedback
has been injected into the context.
*/
func (ctx *Context) Peek() string {
	return ctx.userPrompt + "\n\n" + ctx.cleanedContext.String()
}

/*
Poke is a helper function that allows us to poke the reviewer and prompt engineer
agents, so they can generate feedback and update the context.
*/
func (ctx *Context) Poke(currentContext string) <-chan provider.Event {
	out := make(chan provider.Event)

	go func() {
		defer close(out)

		// Review the output of the last agent.
		ctx.reviewer.buffer.Poke(provider.Message{
			Role:    "assistant",
			Content: ctx.currentContext.String(),
		})

		for event := range ctx.reviewer.Generate() {
			ctx.currentContext.WriteString(event.Content)
			out <- event
		}

		ctx.extractFeedback(ctx.currentContext.String())

		for _, agent := range []*Agent{
			ctx.promptEngineer,
			ctx.vectorMemorizer,
			ctx.graphMemorizer,
		} {
			agent.buffer.Poke(provider.Message{
				Role:    "assistant",
				Content: ctx.currentContext.String(),
			})

			for event := range agent.Generate() {
				ctx.cleanedContext.WriteString(event.Content)
				out <- event
			}
		}

		// Query the memory stores for any relevant information, and inject it into the context.
		ctx.cleanedContext.WriteString("\n")
		ctx.cleanedContext.WriteString("<memories>")

		ctx.vectorQuery.buffer.Poke(provider.Message{
			Role:    "assistant",
			Content: ctx.currentContext.String(),
		})

		for event := range ctx.vectorQuery.Generate() {
			ctx.cleanedContext.WriteString(event.Content)
			out <- event
		}

		ctx.graphQuery.buffer.Poke(provider.Message{
			Role:    "assistant",
			Content: ctx.currentContext.String(),
		})

		for event := range ctx.graphQuery.Generate() {
			ctx.cleanedContext.WriteString(event.Content)
			out <- event
		}

		ctx.cleanedContext.WriteString("</memories>")
		ctx.cleanedContext.WriteString("\n")
	}()

	return out
}

/*
extractFeedback extracts the feedback from the given context, and returns it
as a map, with the agent name as the key, and the feedback as the value.
*/
func (ctx *Context) extractFeedback(response string) {
	blocks := utils.ExtractJSONBlocks(response)

	for _, blockMap := range blocks {
		feedback := blockMap["content"].(string)
		ctx.feedback[blockMap["name"].(string)] = feedback
	}
}
