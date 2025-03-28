package tools

import (
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/trengo"
)

func init() {
	fmt.Println("tools.trengo.init")
	provider.RegisterTool("trengo")
}

type Trengo struct {
	buffer *stream.Buffer
	client *trengo.Client
	Schema *provider.Tool
}

func NewTrengo() *Trengo {
	client := trengo.NewClient()

	return &Trengo{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("trengo.Client.buffer")

			if _, err = io.Copy(client, artifact); err != nil {
				return errnie.Error(err)
			}

			if _, err = io.Copy(artifact, client); err != nil {
				return errnie.Error(err)
			}

			return nil
		}),
		client: client,
		Schema: GetToolSchema("trengo"),
	}
}

func (trengo *Trengo) Read(p []byte) (n int, err error) {
	errnie.Debug("trengo.Read")
	return trengo.buffer.Read(p)
}

func (trengo *Trengo) Write(p []byte) (n int, err error) {
	errnie.Debug("trengo.Write")
	return trengo.buffer.Write(p)
}

func (trengo *Trengo) Close() error {
	errnie.Debug("trengo.Close")
	return trengo.buffer.Close()
}
