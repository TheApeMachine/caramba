package workflow

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
NewFlipFlop creates a bidirectional data flow between two ReadWriters.
It first copies data from the source to the destination, then copies
any response back from the destination to the source. This creates
a complete round-trip data exchange.

Parameters:
  - from: Source ReadWriter to read initial data from and write response to
  - to: Destination ReadWriter to write data to and read response from

Returns:
  - error: Any error that occurred during the data exchange
*/
func NewFlipFlop(from io.ReadWriter, to io.ReadWriter) (err error) {
	errnie.Debug("workflow.NewFlipFlop.Flip")
	if _, err = io.Copy(to, from); err != nil {
		return errnie.Error(err)
	}

	errnie.Debug("workflow.NewFlipFlop.Flop")
	if _, err = io.Copy(from, to); err != nil {
		return errnie.Error(err)
	}

	return nil
}
