package provider

import (
	"context"
	"io"
	"os"
	"testing"

	"github.com/openai/openai-go"
	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func testParams(stream bool) *Params {
	model := openai.ChatModelGPT4oMini
	errnie.Debug("creating test params", "model", model)

	params := &Params{
		Model: model,
		Messages: []*Message{
			{
				Role:    MessageRoleUser,
				Content: "Hello, this is a test message",
			},
		},
		Tools:            nil,
		Temperature:      0.7,
		TopP:             1.0,
		TopK:             40,
		PresencePenalty:  0.0,
		FrequencyPenalty: 0.0,
		MaxTokens:        2048,
		Stream:           stream,
	}

	return params
}

func testEvent(stream bool) *datura.Artifact {
	ctx := testParams(stream)
	return datura.New(
		datura.WithPayload(ctx.Marshal()),
	)
}

func TestNewOpenAIProvider(t *testing.T) {
	Convey("Given a new OpenAI provider", t, func() {
		Convey("When created with explicit API key", func() {
			provider := NewOpenAIProvider(WithAPIKey(os.Getenv("OPENAI_API_KEY")))
			So(provider, ShouldNotBeNil)
			So(provider.client, ShouldNotBeNil)
			So(provider.buffer, ShouldNotBeNil)
			So(provider.params, ShouldNotBeNil)
		})

		Convey("When created with environment API key", func() {
			provider := NewOpenAIProvider(WithAPIKey(os.Getenv("OPENAI_API_KEY")))
			So(provider, ShouldNotBeNil)
			So(provider.client, ShouldNotBeNil)
		})
	})
}

func TestOpenAIProvider_Write(t *testing.T) {
	Convey("Given an OpenAI provider", t, func() {
		provider := NewOpenAIProvider(WithAPIKey(os.Getenv("OPENAI_API_KEY")))

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
		provider := NewOpenAIProvider(WithAPIKey(os.Getenv("OPENAI_API_KEY")))

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
		provider := NewOpenAIProvider(WithAPIKey(os.Getenv("OPENAI_API_KEY")))
		params := &openai.ChatCompletionNewParams{}

		Convey("When building tools with nil context", func() {
			tools := provider.buildTools(params)
			So(tools, ShouldBeEmpty)
		})

		Convey("When building tools with valid context", func() {
			provider.params = &Params{
				Tools: []*Tool{
					{
						Function: Function{
							Name:        "test_tool",
							Description: "A test tool",
							Parameters: Parameters{
								Properties: []Property{
									{
										Name:        "test_param",
										Type:        "string",
										Description: "A test parameter",
									},
								},
								Required: []string{"test_param"},
							},
						},
					},
				},
			}
			err := provider.buildTools(params)
			So(err, ShouldBeNil)
			So(params.Tools, ShouldNotBeEmpty)
		})
	})
}
