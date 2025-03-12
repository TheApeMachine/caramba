package ai

import (
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
ContextData holds the structured data for an AI context.
*/
type ContextData struct {
	Model            string          `json:"model"`
	Messages         []*core.Message `json:"messages"`
	Tools            []*core.Tool    `json:"tools"`
	Process          *core.Process   `json:"process"`
	Temperature      float64         `json:"temperature"`
	TopP             float64         `json:"top_p"`
	TopK             int             `json:"top_k"`
	PresencePenalty  float64         `json:"presence_penalty"`
	FrequencyPenalty float64         `json:"frequency_penalty"`
	MaxTokens        int             `json:"max_tokens"`
	StopSequences    []string        `json:"stop_sequences"`
	Stream           bool            `json:"stream"`
}

/*
Context represents an AI context and implements io.ReadWriteCloser.
*/
type Context struct {
	*ContextData
	*stream.Buffer
}

/*
NewContext creates a new context with default values.
*/
func NewContext() *Context {
	errnie.Debug("ai.NewContext")

	ctx := &Context{
		ContextData: &ContextData{
			Model:            tweaker.GetModel(tweaker.GetProvider()),
			Messages:         []*core.Message{},
			Tools:            []*core.Tool{},
			Process:          nil,
			Temperature:      tweaker.GetTemperature(),
			TopP:             tweaker.GetTopP(),
			TopK:             tweaker.GetTopK(),
			PresencePenalty:  tweaker.GetPresencePenalty(),
			FrequencyPenalty: tweaker.GetFrequencyPenalty(),
			MaxTokens:        tweaker.GetMaxTokens(),
			StopSequences:    tweaker.GetStopSequences(),
			Stream:           tweaker.GetStream(),
		},
	}

	ctx.Buffer = stream.NewBuffer(
		&core.Event{},
		ctx,
		func(event any) error {
			ctx.Messages = append(ctx.Messages, event.(*core.Event).Message)
			return nil
		},
	)

	return ctx
}

/*
Read implements the io.Reader interface for Context.

It flushes the buffer and reads data into the provided byte slice.
Returns the number of bytes read and any error encountered.
*/
func (ctx *Context) Read(p []byte) (n int, err error) {
	errnie.Debug("ai.Context.Read")
	return ctx.Buffer.Read(p)
}

/*
Write implements the io.Writer interface for Context.

It unmarshals incoming data into an Event, adds the Event's message to the
Context's Messages, and encodes the updated Context data into the buffer.
Returns the number of bytes written and any error encountered.
*/
func (ctx *Context) Write(p []byte) (n int, err error) {
	errnie.Debug("ai.Context.Write", "p", string(p))
	return ctx.Buffer.Write(p)
}

/*
Close implements the io.Closer interface for Context.

It flushes the buffer and cleans up resources by setting references to nil.
Returns nil as it doesn't produce errors.
*/
func (ctx *Context) Close() error {
	errnie.Debug("ai.Context.Close")
	return ctx.Buffer.Close()
}
