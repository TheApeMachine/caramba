package memory

import "io"

type Store struct {
	io.ReadWriteCloser
}
