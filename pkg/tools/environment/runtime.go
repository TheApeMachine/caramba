package environment

import (
	"context"
	"io"
)

type Runtime interface {
	CreateContainer(ctx context.Context) error
	StartContainer(ctx context.Context) error
	StopContainer(ctx context.Context) error
	AttachIO(stdin io.Reader, stdout, stderr io.Writer) error
	ExecuteCommand(ctx context.Context, command string, stdout, stderr io.Writer) error
	PullImage(ctx context.Context, ref string) error
	BuildImage(ctx context.Context, dockerfile []byte, tag string) error
}
