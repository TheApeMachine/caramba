package system

import (
	"fmt"
	"sync"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var once sync.Once
var systemInstance *System

func NewSystemTool() (*System, error) {
	var err error

	once.Do(func() {
		systemInstance, err = NewCapnpSystem()
		if err != nil {
			errnie.Error(err)
		}
	})

	return systemInstance, nil
}

// HandleToolCall processes a tool call with the given arguments
func (st *System) HandleToolCall(args map[string]any) (string, error) {
	errnie.Debug("system.HandleToolCall", "args", args)

	cmd, ok := args["command"].(string)
	if !ok {
		return "", errnie.Error(fmt.Errorf("invalid command format"))
	}

	switch cmd {
	case "inspect":
		return st.handleInspect(args)
	default:
		return "Unknown command", errnie.Error(fmt.Errorf("unknown command: %s", cmd))
	}
}

func (st *System) handleInspect(args map[string]any) (string, error) {
	inspectArg, ok := args["inspect_arg"].(string)
	if !ok {
		return "", errnie.Error(fmt.Errorf("invalid inspect argument"))
	}

	switch inspectArg {
	case "agents":
		agents, err := st.Agents()
		if err != nil {
			return "", errnie.Error(err)
		}

		agentList := make([]string, 0, agents.Len())

		for i := range agents.Len() {
			agent := agents.At(i)

			name, err := agent.Name()

			if err != nil {
				errnie.Error(err)
				continue
			}

			agentList = append(agentList, name)
		}

		return fmt.Sprintf("Available agents: %v", agentList), nil

	default:
		return "", errnie.Error(fmt.Errorf("unknown inspect argument: %s", inspectArg))
	}
}

func NewCapnpSystem() (*System, error) {
	var (
		arena  = capnp.SingleSegment(nil)
		seg    *capnp.Segment
		system System
		err    error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil, errnie.Error(err)
	}

	if system, err = NewRootSystem(seg); errnie.Error(err) != nil {
		return nil, errnie.Error(err)
	}

	return &system, nil
}

func (sys *System) AddAgent(id, name string) error {
	agents, err := sys.Agents()

	if errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	// Create a new AgentList, which is +1 in length
	agentList, err := NewAgent_List(sys.Segment(), int32(agents.Len()+1))

	if errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	for i := range agents.Len() {
		agent := agents.At(i)
		// Copy existing agent to the new list
		if err = agentList.Set(i, agent); errnie.Error(err) != nil {
			return errnie.Error(err)
		}
	}

	// Create the new agent
	newAgent, err := NewAgent(sys.Segment())

	if errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	if errnie.Error(newAgent.SetName(name)) != nil {
		return errnie.Error(err)
	}

	// Add the new agent to the end of the new list
	if err = agentList.Set(agents.Len(), newAgent); errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	// Set the updated list back to the system
	if err := sys.SetAgents(agentList); errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	return nil
}
