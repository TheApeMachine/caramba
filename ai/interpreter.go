package ai

import (
	"encoding/json"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/ai/tasks"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

var taskMap = map[string]tasks.Task{
	"ignore":   tasks.NewIgnore(),
	"help":     tasks.NewHelp(),
	"break":    tasks.NewBreak(),
	"recall":   tasks.NewRecall(),
	"remember": tasks.NewRemember(),
	"bash":     tasks.NewBash(),
	"terminal": tasks.NewTerminal(),
	"web":      tasks.NewWeb(),
	"optimize": tasks.NewOptimize(),
}

type Command struct {
	Task tasks.Task
	Args map[string]any
}

/*
Interpreter is an object that extracts and interprets commands from unstructured text.
It maps any commands to handler methods.
*/
type Interpreter struct {
	ctx      *drknow.Context
	commands []Command
}

/*
NewInterpreter creates a new Interpreter.
*/
func NewInterpreter(
	ctx *drknow.Context,
) *Interpreter {
	return &Interpreter{
		ctx:      ctx,
		commands: make([]Command, 0),
	}
}

func (interpreter *Interpreter) Execute() tasks.Bridge {
	var bridge tasks.Bridge

	// Guard against empty commands
	if len(interpreter.commands) == 0 {
		return nil
	}

	// Execute each command safely
	for _, command := range interpreter.commands {
		// Guard against nil task
		if command.Task == nil {
			continue
		}

		// Execute with nil-safe argument handling
		if command.Args == nil {
			command.Args = make(map[string]any)
		}

		bridge = command.Task.Execute(interpreter.ctx, command.Args)
	}

	return bridge
}

func (interpreter *Interpreter) Interpret() (*Interpreter, AgentState) {
	interpreter.commands = make([]Command, 0)
	agentState := AgentStateGenerating

	messages := interpreter.ctx.Identity.Params.Thread.Messages
	if len(messages) == 0 || messages[len(messages)-1].Role != provider.RoleAssistant {
		return interpreter, agentState
	}

	// Extract code blocks.
	blocks := utils.ExtractJSONBlocks(messages[len(messages)-1].Content)
	for _, block := range blocks {
		if tool, ok := block["tool"].(string); ok {
			interpreter.commands = append(interpreter.commands, Command{
				Task: taskMap[tool],
				Args: block,
			})
		}
	}

	// Unmarshal the message content.
	var content map[string]any
	err := json.Unmarshal([]byte(messages[len(messages)-1].Content), &content)
	if err != nil {
		return interpreter, agentState
	}

	return interpreter, agentState
}
