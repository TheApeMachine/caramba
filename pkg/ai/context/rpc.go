package context

import (
	context "context"

	message "github.com/theapemachine/caramba/pkg/ai/message"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ContextRPCServer struct {
	context Context
}

func NewContextRPCServer(context Context) *ContextRPCServer {
	errnie.Trace("context.NewContextRPCServer")

	return &ContextRPCServer{context: context}
}

func ContextToClient(context Context) RPC {
	errnie.Trace("context.ContextToClient")

	server := NewContextRPCServer(context)
	return RPC_ServerToClient(server)
}

func (srv *ContextRPCServer) Add(ctx context.Context, call RPC_add) error {
	errnie.Trace("context.Add")

	msg := errnie.Try(call.Args().Context())
	messages := errnie.Try(srv.context.Messages())

	ml, err := message.NewMessage_List(
		srv.context.Segment(), int32(messages.Len()+1),
	)

	if errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	for i := range messages.Len() {
		ml.Set(i, messages.At(i))
	}

	ml.Set(messages.Len(), msg)

	srv.context.SetMessages(ml)

	return nil
}
