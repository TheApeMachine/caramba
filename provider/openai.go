package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
OpenAI is a provider for the OpenAI API.
*/
type OpenAI struct {
	client *openai.Client
}

/*
NewOpenAI creates a new OpenAI provider.
*/
func NewOpenAI(apiKey string) *OpenAI {
	return &OpenAI{
		client: openai.NewClient(option.WithAPIKey(apiKey)),
	}
}

/*
Stream streams a chat completion.
*/
func (prvdr *OpenAI) Stream(input *datura.Artifact) chan *datura.Artifact {
	errnie.Info("⚡ *OpenAI.Stream")

	out := make(chan *datura.Artifact)

	if prvdr.client == nil {
		errnie.Error(fmt.Errorf("OpenAI client not initialized"))
		close(out)
		return out
	}

	params, err := utils.DecryptPayload(input)
	if err != nil {
		errnie.Error(fmt.Errorf("failed to decrypt payload: %w", err))
		close(out)
		return out
	}

	providerParams := ProviderParams{}
	if err := json.Unmarshal(params, &providerParams); err != nil {
		errnie.Error(fmt.Errorf("failed to unmarshal provider params: %w", err))
		close(out)
		return out
	}

	chatParams := openai.ChatCompletionNewParams{}

	chatParams.Model = openai.F(viper.GetViper().GetString("models.openai"))

	messages := []openai.ChatCompletionMessageParamUnion{}

	for _, message := range providerParams.Messages {
		switch message.Role {
		case "system":
			messages = append(messages, openai.SystemMessage(message.Content))
		case "user":
			messages = append(messages, openai.UserMessage(message.Content))
		case "assistant":
			messages = append(messages, openai.AssistantMessage(message.Content))
		}
	}

	chatParams.Messages = openai.F(messages)

	if providerParams.StructuredResponse != nil {
		schemaParam := openai.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:        openai.F(providerParams.StructuredResponse.Name),
			Description: openai.F(providerParams.StructuredResponse.Description),
			Schema:      openai.F(providerParams.StructuredResponse.Schema),
			Strict:      openai.Bool(true),
		}

		chatParams.ResponseFormat = openai.F[openai.ChatCompletionNewParamsResponseFormatUnion](
			openai.ResponseFormatJSONSchemaParam{
				Type:       openai.F(openai.ResponseFormatJSONSchemaTypeJSONSchema),
				JSONSchema: openai.F(schemaParam),
			},
		)
	}

	if len(providerParams.Tools) > 0 {
		tools := []openai.ChatCompletionToolParam{}

		for _, tool := range providerParams.Tools {
			tools = append(tools, openai.ChatCompletionToolParam{
				Type: openai.F(openai.ChatCompletionToolTypeFunction),
				Function: openai.F(openai.FunctionDefinitionParam{
					Name:        openai.String(tool.Name),
					Description: openai.String(tool.Description),
					Parameters: openai.F(openai.FunctionParameters{
						"type":       "object",
						"properties": tool.Parameters.Properties,
						"required":   tool.Parameters.Required,
					}),
				}),
			})
		}

		chatParams.Tools = openai.F(tools)
	}

	go func() {
		defer close(out)
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		stream := prvdr.client.Chat.Completions.NewStreaming(ctx, chatParams)
		acc := openai.ChatCompletionAccumulator{}

		errnie.Log("\n\n===REQUEST===\n%v\n\n===END===", chatParams)

		for stream.Next() {
			select {
			case <-ctx.Done():
				return
			default:
				chunk := stream.Current()
				acc.AddChunk(chunk)

				if content, ok := acc.JustFinishedContent(); ok {
					errnie.Log("\n\n===RESPONSE===\n%v\n\n===END===", content)
					artifact, err := prvdr.buildArtifact(datura.ArtifactScopeFunctionCall, "\n")
					if err != nil {
						panic(err)
					}
					out <- artifact
					cancel()
				}

				if tool, ok := acc.JustFinishedToolCall(); ok {
					toolJSON, err := json.Marshal(tool)
					if err != nil {
						panic(err)
					}
					artifact, err := prvdr.buildArtifact(datura.ArtifactScopeFunctionCall, string(toolJSON))
					if err != nil {
						panic(err)
					}
					out <- artifact
					cancel()
				}

				if refusal, ok := acc.JustFinishedRefusal(); ok {
					artifact, err := prvdr.buildArtifact(datura.ArtifactScopeChatCompletion, refusal)
					if err != nil {
						panic(err)
					}
					out <- artifact
					cancel()
				}

				// Only send actual content deltas
				if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
					artifact, err := prvdr.buildArtifact(
						datura.ArtifactScopeChatCompletion,
						chunk.Choices[0].Delta.Content,
					)

					if errnie.Error(err) != nil {
						errArtifact := datura.NewArtifactBuilder(
							datura.MediaTypeTextPlain,
							datura.ArtifactRoleAssistant,
							datura.ArtifactScopeChatCompletion,
						)
						errArtifact.SetPayload([]byte(err.Error()))
						artifact, _ = errArtifact.Build()
					}

					out <- artifact
				}

				if err := stream.Err(); err != nil {
					errArtifact := datura.NewArtifactBuilder(
						datura.MediaTypeTextPlain,
						datura.ArtifactRoleAssistant,
						datura.ArtifactScopeChatCompletion,
					)
					errArtifact.SetPayload([]byte(err.Error()))
					artifact, _ := errArtifact.Build()

					out <- artifact
					cancel()
				}
			}
		}
	}()

	return out
}

func (prvdr *OpenAI) buildArtifact(scope datura.ArtifactScope, data string) (*datura.Artifact, error) {
	artifact := datura.NewArtifactBuilder(
		datura.MediaTypeTextPlain,
		datura.ArtifactRoleAssistant,
		scope,
	)

	artifact.SetPayload([]byte(data))
	return artifact.Build()
}
