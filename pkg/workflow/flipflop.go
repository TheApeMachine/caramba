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

This is most commonly used when you want to flip an artifact into
another process, which will set some state on that artifact, that
will be accessible once the process flops the artifact.

Parameters:
  - from: Source ReadWriter to read initial data from and write response to
  - to: Destination ReadWriter to write data to and read response from

Returns:
  - error: Any error that occurred during the data exchange

Example:

```go

	package main

	type Setter struct {
		buffer *stream.Buffer
	}

	func NewSetter() *Setter {
		return &Setter{
			buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
				// Set the output metadata on the artifact.
				artifact.SetMetaValue("output", "hello")
				return nil
			}),
		}
	}

	func (s *Setter) Read(p []byte) (n int, err error) {
		return s.buffer.Read(p)
	}

	func (s *Setter) Write(p []byte) (n int, err error) {
		return s.buffer.Write(p)
	}

	func (s *Setter) Close() error {
		return s.buffer.Close()
	}

	func main() {
		// Create a new setter.
		setter := NewSetter()

		// Create a new, empty artifact.
		artifact := datura.NewArtifact()

		// Flip the artifact into the setter, and flop it back.
		workflow.NewFlipFlop(artifact, setter)

		// Read the output metadata from the artifact.
		fmt.Println(datura.GetMetaValue[string](artifact, "output"))
		// Output: hello
	}

````
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
