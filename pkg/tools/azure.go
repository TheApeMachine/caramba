package tools

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/github"
	"github.com/theapemachine/caramba/pkg/workflow"
)

func init() {
	fmt.Println("tools.azure.init")
	provider.RegisterTool("azure")
}

/*
Azure represents the Azure DevOps integration tool that provides functionality
for interacting with Azure DevOps services. It implements io.ReadWriteCloser
interface for streaming data through the tool.
*/
type Azure struct {
	buffer *stream.Buffer
	client *github.Client
	Schema *provider.Tool
}

/*
NewAzure creates and initializes a new Azure tool instance with a configured
GitHub client and buffer for handling data streams.

Returns a new Azure tool instance ready for use.
*/
func NewAzure() *Azure {
	client := github.NewClient()

	return &Azure{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("azure.Client.buffer")

			if err = workflow.NewFlipFlop(artifact, client); err != nil {
				return errnie.Error(err)
			}

			return nil
		}),
		client: client,
		Schema: GetToolSchema("azure"),
	}
}

/*
Read implements the io.Reader interface.

It reads data from the internal buffer.
Returns the number of bytes read and any error encountered.
*/
func (azure *Azure) Read(p []byte) (n int, err error) {
	errnie.Debug("azure.Read")
	return azure.buffer.Read(p)
}

/*
Write implements the io.Writer interface.

It writes data to the internal buffer.
Returns the number of bytes written and any error encountered.
*/
func (azure *Azure) Write(p []byte) (n int, err error) {
	errnie.Debug("azure.Write")
	return azure.buffer.Write(p)
}

/*
Close implements the io.Closer interface.

It closes the internal buffer and cleans up resources.
Returns any error encountered during closing.
*/
func (azure *Azure) Close() error {
	errnie.Debug("azure.Close")
	return azure.buffer.Close()
}
