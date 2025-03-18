package workflow

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Feedback is a wrapper that allows a workflow to send data both forwards and backwards
in a pipeline. This is useful for responses generated by the LLM provider, which need
to flow back to the agent, so it can store it in the current context.
*/
type Feedback struct {
	forward  io.ReadWriter
	backward io.Writer
	tee      io.Reader
}

func NewFeedback(forward io.ReadWriter, backward io.Writer) *Feedback {
	return &Feedback{
		forward:  forward,
		backward: backward,
		tee:      io.TeeReader(forward, backward),
	}
}

func (feedback *Feedback) Read(p []byte) (n int, err error) {
	errnie.Debug("feedback.Read")
	return feedback.tee.Read(p)
}

func (feedback *Feedback) Write(p []byte) (n int, err error) {
	errnie.Debug("feedback.Write")
	// Reset the tee with the updated forward component after writing
	if n, err = feedback.forward.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	return n, nil
}

func (feedback *Feedback) Close() error {
	errnie.Debug("feedback.Close")

	// Close the forward component if it implements io.Closer
	if closer, ok := feedback.forward.(io.Closer); ok {
		if err := closer.Close(); err != nil {
			return errnie.Error(err)
		}
	}

	// Close the backward component if it implements io.Closer
	if closer, ok := feedback.backward.(io.Closer); ok {
		if err := closer.Close(); err != nil {
			return errnie.Error(err)
		}
	}

	return nil
}
