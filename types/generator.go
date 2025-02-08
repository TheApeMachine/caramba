package types

import "io"

type Generator interface {
	io.ReadWriteCloser
}
