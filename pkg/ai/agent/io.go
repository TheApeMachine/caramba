package agent

import (
	"io"

	capnp "capnproto.org/go/capnp/v3"
	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type AgentState uint

const (
	AgentStateUninitialized AgentState = iota
	AgentStateInitialized
	AgentStateBuffered
)

/*
Read implements the io.Reader interface for the Agent.
It streams the agent using a Cap'n Proto Encoder.
*/
func (agent *Agent) Read(p []byte) (n int, err error) {
	errnie.Trace("agent.Read")

	builder := datura.NewRegistry().Get(agent)

	if agent.Is(errnie.StateReady) {
		// Buffer is empty, encode current message state
		if err = builder.Encoder.Encode(agent.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		if err = builder.Buffer.Flush(); err != nil {
			return 0, errnie.Error(err)
		}

		agent.ToState(errnie.StateBusy)
	}

	return builder.Buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Agent.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (agent *Agent) Write(p []byte) (n int, err error) {
	errnie.Trace("agent.Write")

	builder := datura.NewRegistry().Get(agent)

	if len(p) == 0 {
		return 0, nil
	}

	if n, err = builder.Buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	if err = builder.Buffer.Flush(); err != nil {
		return n, errnie.Error(err)
	}

	var (
		msg *capnp.Message
		buf Agent
	)

	if msg, err = builder.Decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if buf, err = ReadRootAgent(msg); err != nil {
		return n, errnie.Error(err)
	}

	agent = &buf
	agent.ToState(errnie.StateReady)
	return n, nil
}

/*
Close implements the io.Closer interface for the Agent.
*/
func (agent *Agent) Close() error {
	errnie.Trace("agent.Close")

	builder := datura.NewRegistry().Get(agent)

	if err := builder.Buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	builder.Buffer = nil
	builder.Encoder = nil
	builder.Decoder = nil
	datura.NewRegistry().Unregister(agent)
	return nil
}
