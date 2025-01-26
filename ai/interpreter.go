package ai

import (
	"regexp"
	"strings"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/ai/tasks"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/errnie"
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
	"web":      tasks.NewWeb(),
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
	ctx         *drknow.Context
	commands    []Command
	accumulator *stream.Accumulator
}

/*
NewInterpreter creates a new Interpreter.
*/
func NewInterpreter(
	ctx *drknow.Context,
	accumulator *stream.Accumulator,
) *Interpreter {
	return &Interpreter{
		ctx:         ctx,
		commands:    make([]Command, 0),
		accumulator: accumulator,
	}
}

func (interpreter *Interpreter) Execute() {
	for _, command := range interpreter.commands {
		command.Task.Execute(interpreter.ctx, interpreter.accumulator, command.Args)
	}
}

func (interpreter *Interpreter) Interpret() *Interpreter {
	// This regex matches: <command param1="value1" param2=[value2]>
	regexpattern := regexp.MustCompile(`<(\w+)(?:\s+(\w+)\s*=\s*(?:"([^"]*)"|(\[[^\]]*\])))*>`)
	matches := regexpattern.FindAllStringSubmatch(interpreter.accumulator.String(), -1)

	for _, match := range matches {
		command := strings.ToLower(match[1])
		args := make(map[string]any)

		// The full match is at index 0, command name at index 1
		// After that, every group of 3 elements represents: key, quoted value, array value
		for i := 2; i < len(match); i += 3 {
			if match[i] != "" { // If we have a parameter name
				key := match[i]
				// Value could be either quoted string or array
				value := match[i+1]
				if value == "" {
					value = match[i+2] // Use array value if quoted string is empty
				}
				if value != "" {
					args[key] = value
				}
			}
		}

		if _, ok := taskMap[command]; !ok {
			errnie.Warn("unknown command: %s", command)
			continue
		}

		interpreter.commands = append(interpreter.commands, Command{
			Task: taskMap[command],
			Args: args,
		})
	}

	return interpreter
}
