package provider

import (
	"bytes"
	"context"
	"io"
	"os"
	"testing"

	"github.com/openai/openai-go"
	. "github.com/smartystreets/goconvey/convey"
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
)

func testParams(stream bool) *aiCtx.Artifact {
	model := openai.ChatModelGPT4oMini
	errnie.Debug("creating test params", "model", model)

	ctx := aiCtx.New(
		model,
		nil,    // messages
		nil,    // tools
		nil,    // process
		0.7,    // temperature
		1.0,    // topP
		40,     // topK
		0.0,    // presencePenalty
		0.0,    // frequencyPenalty
		2048,   // maxTokens
		stream, // stream
	)

	// Create a message event and add it to the context
	msgEvt := event.New(
		"test.pipeline",
		event.MessageEvent,
		event.UserRole,
		message.New(
			message.UserRole,
			"test-name",
			"test message",
		).Marshal(),
	)
	msgEvt.ToContext(ctx)

	return ctx
}

func testEvent(stream bool) *event.Artifact {
	ctx := testParams(stream)
	return event.New(
		"test.pipeline",
		event.ContextEvent,
		event.UserRole,
		ctx.Marshal(),
	)
}

func TestNewOpenAIProvider(t *testing.T) {
	Convey("Given a new OpenAI provider", t, func() {
		Convey("When created with explicit API key", func() {
			provider := NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "")
			So(provider, ShouldNotBeNil)
			So(provider.client, ShouldNotBeNil)
			So(provider.buffer, ShouldNotBeNil)
			So(provider.params, ShouldNotBeNil)
		})

		Convey("When created with environment API key", func() {
			provider := NewOpenAIProvider("", "")
			So(provider, ShouldNotBeNil)
			So(provider.client, ShouldNotBeNil)
		})
	})
}

func TestOpenAIProvider_Read(t *testing.T) {
	Convey("Given an OpenAI provider with data in buffer", t, func() {
		provider := NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "")

		Convey("When reading in non-streaming mode", func() {
			testEvent := testEvent(false)
			errChan := make(chan error, 1)
			nChan := make(chan int, 1)

			// Write to the provider
			go func() {
				n, err := io.Copy(provider, testEvent)
				errChan <- err
				nChan <- int(n)
			}()

			// Read from the buffer
			buf := bytes.NewBuffer([]byte{})
			n, err := io.Copy(buf, provider)

			// Check the write results
			writeErr := <-errChan
			writeN := <-nChan

			So(writeErr, ShouldBeNil)
			So(writeN, ShouldBeGreaterThan, 0)

			// Check the read results
			So(err, ShouldBeNil)
			So(n, ShouldBeGreaterThan, 0)
		})

		Convey("When reading in streaming mode", func() {
			testEvent := testEvent(true)
			errChan := make(chan error, 1)
			nChan := make(chan int, 1)

			// Write to the provider
			go func() {
				n, err := io.Copy(provider, testEvent)
				errChan <- err
				nChan <- int(n)
			}()

			// Read from the buffer
			buf := bytes.NewBuffer([]byte{})
			n, err := io.Copy(buf, provider)

			// Check the write results
			writeErr := <-errChan
			writeN := <-nChan

			So(writeErr, ShouldBeNil)
			So(writeN, ShouldBeGreaterThan, 0)

			// Check the read results
			So(err, ShouldBeNil)
			So(n, ShouldBeGreaterThan, 0)
		})
	})
}

func TestOpenAIProvider_Write(t *testing.T) {
	Convey("Given an OpenAI provider", t, func() {
		provider := NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "")

		Convey("When writing a message", func() {
			testEvent := testEvent(false)
			n, err := io.Copy(provider, testEvent)
			So(err, ShouldBeNil)
			So(n, ShouldBeGreaterThan, 0)
		})

		Convey("When writing with streaming enabled", func() {
			testEvent := testEvent(true)

			n, err := io.Copy(provider, testEvent)
			So(err, ShouldBeNil)
			So(n, ShouldBeGreaterThan, 0)

			n, err = io.Copy(provider, testEvent)
			So(err, ShouldBeNil)
			So(n, ShouldBeGreaterThan, 0)
		})
	})
}

func TestOpenAIProvider_Close(t *testing.T) {
	Convey("Given an OpenAI provider", t, func() {
		provider := NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "")

		Convey("When closing the provider", func() {
			err := provider.Close()
			So(err, ShouldBeNil)
		})

		Convey("When closing with active context", func() {
			ctx, cancel := context.WithCancel(context.Background())
			provider.ctx = ctx
			provider.cancel = cancel
			err := provider.Close()
			So(err, ShouldBeNil)
		})
	})
}

func TestOpenAIProvider_BuildTools(t *testing.T) {
	Convey("Given an OpenAI provider", t, func() {
		provider := NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "")
		params := &openai.ChatCompletionNewParams{}

		Convey("When building tools with nil context", func() {
			tools := provider.buildTools(nil, params)
			So(tools, ShouldBeEmpty)
		})

		Convey("When building tools with valid context", func() {
			testEvent := testParams(false)
			tools := provider.buildTools(testEvent, params)
			So(tools, ShouldNotBeNil)
		})
	})
}
