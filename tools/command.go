package tools

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/system"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

type CommandTool struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  provider.Parameter `json:"parameters"`
}

func NewCommandTool() *CommandTool {
	return &CommandTool{
		Name:        "command",
		Description: "This tool is used to execute commands.",
		Parameters: provider.Parameter{
			Properties: map[string]interface{}{
				"command": map[string]interface{}{
					"type":        "string",
					"description": "The command to execute",
					"enum":        []string{"inspect"},
				},
				"args": map[string]interface{}{
					"type":        "string",
					"description": "The arguments for the command",
					"enum":        []string{"system", "agents", "topics"},
				},
			},
			Required: []string{"command", "args"},
		},
	}
}

func (tool *CommandTool) Convert() provider.Tool {
	return provider.Tool{
		Name:        tool.Name,
		Description: tool.Description,
		Parameters:  tool.Parameters,
	}
}

func (tool *CommandTool) Use(agent *ai.Agent, artifact *datura.Artifact) {
	errnie.Info("🔨 *CommandTool.Use")

	decrypted, err := utils.DecryptPayload(artifact)
	if err != nil {
		panic(err)
	}

	var params struct {
		Command string `json:"command"`
		Args    string `json:"args"`
	}

	if err := json.Unmarshal(decrypted, &params); err != nil {
		panic(err)
	}

	errnie.Info("🔨 *CommandTool.Use")

	if params.Command == "inspect" {
		switch params.Args {
		case "system":
			topics := system.NewQueue().GetAllTopics()

			out := []string{
				"<system>",
			}

			for topic, agents := range topics {
				out = append(out, "\t<topic id="+topic+">")
				for _, topicAgent := range agents {
					out = append(out, "\t\t<agent id="+topicAgent.Identity.ID+">")
					out = append(out, "\t\t\t<name>"+topicAgent.Identity.Name+"</name>")
					out = append(out, "\t\t\t<role>"+topicAgent.Identity.Role+"</role>")
					out = append(out, "\t\t</agent>")
				}
				out = append(out, "\t</topic>")
			}

			str := strings.Join(out, "\n")
			fmt.Println(str)

			agent.AddContext(str)
		case "agents":
			agents := system.NewQueue().GetAllAgents()

			out := []string{
				"<agents>",
			}

			for _, agent := range agents {
				out = append(out, "\t<agent id="+agent.Identity.ID+">")
				out = append(out, "\t\t<name>"+agent.Identity.Name+"</name>")
				out = append(out, "\t\t<role>"+agent.Identity.Role+"</role>")
				out = append(out, "\t</agent>")
			}

			out = append(out, "</agents>")
			str := strings.Join(out, "\n")
			fmt.Println(str)

			agent.AddContext(str)
		case "topics":
			topics := system.NewQueue().GetAllTopics()

			out := []string{
				"<topics>",
			}

			for topic := range topics {
				out = append(out, "\t<topic>"+topic+"</topic>")
			}

			out = append(out, "</topics>")
			str := strings.Join(out, "\n")
			fmt.Println(str)

			agent.AddContext(str)
		}
	}
}
