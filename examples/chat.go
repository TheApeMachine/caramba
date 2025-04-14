package examples

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/service/client"
	"github.com/theapemachine/caramba/pkg/task"
)

type ChatExample struct {
	name string
}

func NewChatExample(name string) *ChatExample {
	return &ChatExample{
		name: name,
	}
}

func (example *ChatExample) Run() error {
	errnie.Info("Terminal Chat Agent with A2A/MCP Integration")
	errnie.Info("Type 'exit' to quit or 'help' for commands")

	// Connect to RPC server on port 3211 (3210 + 1)
	errnie.Info("Connecting to RPC server...")
	client, err := client.NewRPCClient(
		client.WithBaseURL("localhost:3211"),
	)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	var (
		request = task.NewTaskRequest(task.NewTask(
			task.WithMessages(
				task.NewSystemMessage("You are a helpful assistant"),
			),
		))
		reader = bufio.NewReader(os.Stdin)
	)

	for {
		fmt.Print("$" + example.name + "> ")
		input, err := reader.ReadString('\n')
		if err != nil {
			return errnie.New(errnie.WithError(err))
		}

		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}

		if input == "exit" {
			break
		}

		if input == "help" {
			example.printHelp()
			continue
		}

		// Add user message to history
		userMsg := task.NewUserMessage(example.name, input)
		request.Params.History = append(request.Params.History, userMsg)

		// Use the streaming approach with SendTaskStream
		responseChan, err := client.SendTaskStream(request)
		if err != nil {
			return errnie.New(errnie.WithError(err))
			// Remove the user message if the send failed
			if len(request.Params.History) > 0 {
				request.Params.History = request.Params.History[:len(request.Params.History)-1]
			}
			continue
		}

		fmt.Print("Assistant: ")

		var (
			responseComplete bool
			assistantMessage task.Message
			assistantText    string
		)

		// Process streaming responses as they arrive
		for response := range responseChan {
			if response.Error != nil {
				return errnie.New(errnie.WithError(response.Error))
			}

			// Check for task completion
			if response.Result != nil && response.Result.Status.State == task.TaskStateCompleted {
				responseComplete = true
				continue
			}

			// Get the message content from the response artifacts or history
			if response.Result != nil && len(response.Result.History) > 0 {
				// If we have a complete history, use the last message
				lastMsg := response.Result.History[len(response.Result.History)-1]

				// Only process assistant messages
				if lastMsg.Role == task.RoleAgent || lastMsg.Role == task.RoleAssistant {
					assistantMessage = lastMsg

					// Extract text from the message parts
					for _, part := range lastMsg.Parts {
						if textPart, ok := part.(*task.TextPart); ok {
							// Print only the newly added text
							newText := textPart.Text
							if len(newText) > len(assistantText) {
								diffText := newText[len(assistantText):]
								fmt.Print(diffText)
								assistantText = newText
							}
							break
						}
					}
				}
			} else if response.Result != nil && len(response.Result.Artifacts) > 0 {
				// Alternative: we might receive artifacts directly
				for _, artifact := range response.Result.Artifacts {
					for _, part := range artifact.Parts {
						if textPart, ok := part.(*task.TextPart); ok {
							fmt.Print(textPart.Text)

							// Add to our accumulated text
							assistantText += textPart.Text

							// Create/update assistant message
							if assistantMessage.Parts == nil {
								assistantMessage = task.Message{
									Role: task.RoleAssistant,
									Parts: []task.Part{
										&task.TextPart{
											Type: "text",
											Text: assistantText,
										},
									},
								}
							} else {
								// Update the existing message's text
								for i, part := range assistantMessage.Parts {
									if textPart, ok := part.(*task.TextPart); ok {
										textPart.Text = assistantText
										assistantMessage.Parts[i] = textPart
										break
									}
								}
							}
							break
						}
					}
				}
			}
		}

		fmt.Println() // End line after response

		// If we received a complete response, add it to history
		if responseComplete && assistantMessage.Parts != nil {
			request.Params.History = append(request.Params.History, assistantMessage)
		} else {
			errnie.Warn("Response may be incomplete")
		}
	}

	return nil
}

func (example *ChatExample) printHelp() {
	fmt.Println("Available commands:")
	fmt.Println("  help  - Show this help message")
	fmt.Println("  exit  - Exit the chat")
	fmt.Println("")
	fmt.Println("Just type your message and press Enter to chat with the assistant.")
}
