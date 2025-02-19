package provider

import (
	"context"
	"encoding/json"

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
	out := make(chan *datura.Artifact)

	params, err := utils.DecryptPayload(input)

	if err != nil {
		panic(err)
	}

	providerParams := ProviderParams{}
	err = json.Unmarshal(params, &providerParams)

	if err != nil {
		panic(err)
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

		stream := prvdr.client.Chat.Completions.NewStreaming(context.Background(), chatParams)
		acc := openai.ChatCompletionAccumulator{}

		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)

			// When this fires, the current chunk value will not contain content data
			if content, ok := acc.JustFinishedContent(); ok {
				errnie.Log("%v\n\n%v", chatParams, content)
				continue
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
				continue
			}

			if refusal, ok := acc.JustFinishedRefusal(); ok {
				artifact, err := prvdr.buildArtifact(datura.ArtifactScopeChatCompletion, refusal)
				if err != nil {
					panic(err)
				}
				out <- artifact
				continue
			}

			// Only send actual content deltas
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				artifact, err := prvdr.buildArtifact(
					datura.ArtifactScopeChatCompletion,
					chunk.Choices[0].Delta.Content,
				)
				if err != nil {
					panic(err)
				}
				out <- artifact
			}
		}

		if err := stream.Err(); err != nil {
			panic(err)
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
