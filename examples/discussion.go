package examples

import (
	"io"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Discussion struct {
	workflow io.ReadWriteCloser
}

func NewDiscussion() *Discussion {
	errnie.Debug("examples.NewDiscussion")

	return &Discussion{
		workflow: workflow.NewRing([]io.ReadWriteCloser{
			&Pipeline{
				workflow: workflow.NewPipeline(
					ai.NewAgent(),
					provider.NewOpenAIProvider("", ""),
					core.NewConverter(),
				),
			},
			&Pipeline{
				workflow: workflow.NewPipeline(
					ai.NewAgent(),
					provider.NewAnthropicProvider("", ""),
					core.NewConverter(),
				),
			},
		}),
	}
}

func (discussion *Discussion) Read(p []byte) (n int, err error) {
	errnie.Debug("examples.Discussion.Read")

	if n, err = discussion.workflow.Read(p); err != nil {
		errnie.NewErrIO(err)
	}

	return n, err
}

func (discussion *Discussion) Write(p []byte) (n int, err error) {
	errnie.Debug("examples.Discussion.Write", "p", string(p))

	if n, err = discussion.workflow.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	return n, err
}

func (discussion *Discussion) Close() error {
	errnie.Debug("examples.Discussion.Close")
	return discussion.workflow.Close()
}
