package ai

import (
	"fmt"
	"strconv"
	"strings"

	provider "github.com/theapemachine/caramba/pkg/api/provider"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func (agent *Agent) InspectAgents(agentName, id string) {
	out := strings.Builder{}
	out.WriteString("INSPECTING SYSTEM\n")

	for _, targetAgent := range Agents {
		targetName, err := targetAgent.Name()

		if err != nil {
			agent.Error(err, id)
			continue
		}

		out.WriteString(targetName + " (agent)\n")
	}

	agent.AddMessage(
		"tool",
		id,
		out.String(),
	)

	fmt.Printf("[%s]\n\n%s\n\n", agentName, out.String())
}

func (agent *Agent) Optimize(
	args map[string]any, toolId string,
) *provider.ProviderParams {
	errnie.Debug("ai.agent.Optimize")

	context, err := agent.Context()

	if err != nil {
		agent.Error(err, toolId)
		return &context
	}

	agentName, err := agent.Name()

	if err != nil {
		agent.Error(err, toolId)
		return &context
	}

	temp := context.Temperature()
	topP := context.TopP()
	topK := context.TopK()
	frequencyPenalty := context.FrequencyPenalty()
	presencePenalty := context.PresencePenalty()

	out := strings.Builder{}

	// Default to "inspect" operation if not specified
	operation := "inspect"
	if op, ok := args["operation"].(string); ok {
		operation = op
	}

	// Check for all possible parameter names (case variations)
	checkTemp := func() (float64, bool) {
		if val, ok := args["temperature"].(float64); ok {
			return val, true
		}
		if val, ok := args["Temperature"].(float64); ok {
			return val, true
		}
		return 0, false
	}

	checkTopP := func() (float64, bool) {
		if val, ok := args["topp"].(float64); ok {
			return val, true
		}
		if val, ok := args["topP"].(float64); ok {
			return val, true
		}
		if val, ok := args["TopP"].(float64); ok {
			return val, true
		}
		return 0, false
	}

	checkTopK := func() (float64, bool) {
		if val, ok := args["topk"].(float64); ok {
			return val, true
		}
		if val, ok := args["topK"].(float64); ok {
			return val, true
		}
		if val, ok := args["TopK"].(float64); ok {
			return val, true
		}
		return 0, false
	}

	checkFreqPenalty := func() (float64, bool) {
		if val, ok := args["frequencypenalty"].(float64); ok {
			return val, true
		}
		if val, ok := args["frequencyPenalty"].(float64); ok {
			return val, true
		}
		if val, ok := args["FrequencyPenalty"].(float64); ok {
			return val, true
		}
		return 0, false
	}

	checkPresPenalty := func() (float64, bool) {
		if val, ok := args["presencepenalty"].(float64); ok {
			return val, true
		}
		if val, ok := args["presencePenalty"].(float64); ok {
			return val, true
		}
		if val, ok := args["PresencePenalty"].(float64); ok {
			return val, true
		}
		return 0, false
	}

	// First, check if any parameter values are provided, regardless of operation
	valuesProvided := false
	
	// Check for system prompt
	systemPromptProvided := false
	var systemPrompt string
	if sp, ok := args["system_prompt"].(string); ok {
		systemPromptProvided = true
		systemPrompt = sp
		valuesProvided = true
	}
	
	// Check for other parameters
	newTemp, tempProvided := checkTemp()
	newTopP, topPProvided := checkTopP()
	newTopK, topKProvided := checkTopK()
	newFrequencyPenalty, freqPenaltyProvided := checkFreqPenalty()
	newPresencePenalty, presPenaltyProvided := checkPresPenalty()
	
	if tempProvided || topPProvided || topKProvided || freqPenaltyProvided || presPenaltyProvided {
		valuesProvided = true
	}

	// If values are provided, apply them regardless of operation
	if valuesProvided {
		// If operation is "inspect" but values are provided, change to "optimize"
		if operation == "inspect" {
			operation = "optimize"
		}
	}

	// Now handle the operation
	switch operation {
	case "inspect":
		out.WriteString("YOUR CURRENT VALUES\n")
		out.WriteString("Temperature: " + strconv.FormatFloat(temp, 'f', -1, 64) + "\n")
		out.WriteString("Top P: " + strconv.FormatFloat(topP, 'f', -1, 64) + "\n")
		out.WriteString("Top K: " + strconv.FormatFloat(topK, 'f', -1, 64) + "\n")
		out.WriteString("Frequency Penalty: " + strconv.FormatFloat(frequencyPenalty, 'f', -1, 64) + "\n")
		out.WriteString("Presence Penalty: " + strconv.FormatFloat(presencePenalty, 'f', -1, 64) + "\n")
	case "optimize":
		if systemPromptProvided {
			out.WriteString("[SYSTEM PROMPT] from: " + systemPrompt + "\n")
			message, err := context.Messages()

			if err != nil {
				agent.Error(err, toolId)
				return &context
			}

			msg, err := provider.NewMessage(agent.Segment())

			if err != nil {
				agent.Error(err, toolId)
				return &context
			}

			if err = msg.SetRole("system"); err != nil {
				agent.Error(err, toolId)
				return &context
			}

			if err = msg.SetContent(systemPrompt); err != nil {
				agent.Error(err, toolId)
				return &context
			}

			message.Set(0, msg)
		}

		if tempProvided {
			out.WriteString("[TEMPERATURE] from: " + strconv.FormatFloat(temp, 'f', -1, 64) + " to: " + strconv.FormatFloat(newTemp, 'f', -1, 64) + "\n")
			context.SetTemperature(newTemp)
		}

		if topPProvided {
			out.WriteString("[TOP P] from: " + strconv.FormatFloat(topP, 'f', -1, 64) + " to: " + strconv.FormatFloat(newTopP, 'f', -1, 64) + "\n")
			context.SetTopP(newTopP)
		}

		if topKProvided {
			out.WriteString("[TOP K] from: " + strconv.FormatFloat(topK, 'f', -1, 64) + " to: " + strconv.FormatFloat(newTopK, 'f', -1, 64) + "\n")
			context.SetTopK(newTopK)
		}

		if freqPenaltyProvided {
			out.WriteString("[FREQUENCY PENALTY] from: " + strconv.FormatFloat(frequencyPenalty, 'f', -1, 64) + " to: " + strconv.FormatFloat(newFrequencyPenalty, 'f', -1, 64) + "\n")
			context.SetFrequencyPenalty(newFrequencyPenalty)
		}

		if presPenaltyProvided {
			out.WriteString("[PRESENCE PENALTY] from: " + strconv.FormatFloat(presencePenalty, 'f', -1, 64) + " to: " + strconv.FormatFloat(newPresencePenalty, 'f', -1, 64) + "\n")
			context.SetPresencePenalty(newPresencePenalty)
		}
	default:
		out.WriteString("Unknown operation: " + operation + "\n")
	}

	// Always add a tool response message, even if the output is empty
	if out.Len() == 0 {
		out.WriteString("Operation completed successfully")
	}

	// Ensure we always add a tool response message
	agent.AddMessage(
		"tool",
		toolId,
		out.String(),
	)

	fmt.Printf("[%s]\n\n%s\n\n", agentName, out.String())

	return &context
}
