package ai

import (
	"bufio"
	"bytes"
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
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
	buffer *bufio.ReadWriter
	dec    *json.Decoder
	enc    *json.Encoder
}

/*
NewContext creates a new context with default values.
*/
func NewContext() *Context {
	errnie.Debug("ai.NewContext")

	buf := bytes.NewBuffer([]byte{})
	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	ctx := &Context{
		ContextData: &ContextData{
			Model:            "gpt-4o",
			Messages:         []*core.Message{},
			Tools:            []*core.Tool{},
			Process:          nil,
			Temperature:      0.7,
			TopP:             1.0,
			TopK:             40,
			PresencePenalty:  0.0,
			FrequencyPenalty: 0.0,
			MaxTokens:        1024,
			StopSequences:    []string{},
			Stream:           true,
		},
		buffer: buffer,
		dec:    json.NewDecoder(buffer),
		enc:    json.NewEncoder(buffer),
	}

	return ctx
}

/*
Read implements io.Reader for Context.
*/
func (ctx *Context) Read(p []byte) (n int, err error) {
	errnie.Debug("ai.Context.Read")

	if err = ctx.buffer.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if n, err = ctx.buffer.Read(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("ai.Context.Read", "n", n, "err", err)

	return n, err
}

// Remove any concurrency logic in Write
func (ctx *Context) Write(p []byte) (n int, err error) {
	errnie.Debug("ai.Context.Write", "p", string(p))

	event := &core.Event{}

	if err = json.Unmarshal(p, event); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	ctx.Messages = append(ctx.Messages, event.Message)
	ctx.buffer.Discard(ctx.buffer.Available())

	if err = ctx.enc.Encode(ctx.ContextData); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	errnie.Debug("ai.Context.Write", "n", n, "err", err)

	return len(p), nil
}

// Remove startStreaming (if it existed) or done channel logic
func (ctx *Context) Close() error {
	errnie.Debug("ai.Context.Close")

	ctx.buffer.Flush()
	ctx.ContextData = nil
	ctx.dec = nil
	ctx.enc = nil

	return nil
}
