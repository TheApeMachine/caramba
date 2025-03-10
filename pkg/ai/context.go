package ai

import (
	"bytes"
	"encoding/json"
	"io"

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
	enc *json.Encoder
	dec *json.Decoder
	in  *bytes.Buffer
	out *bytes.Buffer
}

/*
NewContext creates a new context with default settings.
*/
func NewContext() *Context {
	errnie.Debug("NewContext")

	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	ctx := &Context{
		ContextData: &ContextData{
			Model:       "gpt-4o-mini",
			Messages:    []*core.Message{},
			Tools:       []*core.Tool{},
			Temperature: 0.5,
			TopP:        1.0,
			Stream:      true,
		},
		enc: json.NewEncoder(out),
		dec: json.NewDecoder(in),
		in:  in,
		out: out,
	}

	// Pre-encode the context to JSON for reading
	ctx.enc.Encode(ctx.ContextData)

	return ctx
}

/*
Read implements io.Reader for Context.

It reads from the internal buffer containing the JSON representation.
*/
func (ctx *Context) Read(p []byte) (n int, err error) {
	errnie.Debug("Context.Read")

	if n, err = ctx.out.Read(p); n == 0 {
		return n, io.EOF
	}

	return n, errnie.NewErrIO(err)
}

/*
Write implements io.Writer for Context.

It updates the context based on incoming data.
*/
func (ctx *Context) Write(p []byte) (n int, err error) {
	errnie.Debug("Context.Write", "p", string(p))

	// Reset the output buffer whenever we write new data
	if ctx.out.Len() > 0 {
		ctx.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = ctx.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf ContextData
	if decErr := ctx.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		ctx.ContextData.Model = buf.Model
		ctx.ContextData.Messages = buf.Messages
		ctx.ContextData.Tools = buf.Tools
		ctx.ContextData.Temperature = buf.Temperature
		ctx.ContextData.TopP = buf.TopP
		ctx.ContextData.TopK = buf.TopK
		ctx.ContextData.PresencePenalty = buf.PresencePenalty
		ctx.ContextData.FrequencyPenalty = buf.FrequencyPenalty
		ctx.ContextData.MaxTokens = buf.MaxTokens

		// Re-encode to the output buffer for subsequent reads
		if encErr := ctx.enc.Encode(ctx.ContextData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
}

/*
Close implements io.Closer for Context.
*/
func (ctx *Context) Close() error {
	errnie.Debug("Context.Close")
	ctx.in.Reset()
	ctx.out.Reset()
	return nil
}
