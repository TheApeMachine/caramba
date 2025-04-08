package provider

import (
	"io"

	"capnproto.org/go/capnp/v3"
	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func (prvdr Provider) Read(p []byte) (n int, err error) {
	errnie.Trace("provider.Read")

	builder := datura.NewRegistry().Get(prvdr.ID())

	if prvdr.Is(errnie.StateReady) {
		// Buffer is empty, encode current message state
		if err = builder.Encoder.Encode(prvdr.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		if err = builder.Buffer.Flush(); err != nil {
			return 0, errnie.Error(err)
		}

		prvdr.ToState(errnie.StateBusy)
	}

	return builder.Buffer.Read(p)
}

func (prvdr Provider) Write(p []byte) (n int, err error) {
	errnie.Trace("provider.Write")

	builder := datura.NewRegistry().Get(prvdr.ID())

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
	)

	if msg, err = builder.Decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if prvdr, err = ReadRootProvider(msg); err != nil {
		return n, errnie.Error(err)
	}

	prvdr.ToState(errnie.StateReady)
	return n, nil
}

func (prvdr Provider) Close() error {
	errnie.Trace("provider.Close")

	builder := datura.NewRegistry().Get(prvdr.ID())

	builder.Buffer = nil
	builder.Encoder = nil
	builder.Decoder = nil

	return nil
}
