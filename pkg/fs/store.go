package fs

import (
	"bytes"
	"embed"
	"io"
	"sync"

	"github.com/spf13/afero"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

var once sync.Once
var store *Store

/*
Store represents a file system store with an in-memory filesystem and buffered operations.
*/
type Store struct {
	buffer *stream.Buffer
	conn   *Conn
}

/*
NewStore creates a new file system store with an in-memory filesystem.
It uses a singleton pattern to ensure only one store instance exists.
*/
func NewStore() *Store {
	errnie.Debug("fs.Store.New")
	conn := NewConn()

	once.Do(func() {
		store = &Store{
			buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
				errnie.Debug("fs.Store.buffer")

				var (
					fh    afero.File
					files []string
					path  = datura.GetMetaValue[string](artifact, "path")
					n     int64
				)

				switch artifact.Role() {
				case uint32(datura.ArtifactRoleOpenFile):
					errnie.Debug("fs.Store.buffer", "op", "open", "path", path)
					if fh, err = conn.Open(path); err != nil {
						return errnie.Error(err)
					}
				case uint32(datura.ArtifactRoleSaveFile):
					errnie.Debug("fs.Store.buffer", "op", "save", "path", path)
					if fh, err = conn.Open(path); err != nil {
						return errnie.Error(err)
					}
				case uint32(datura.ArtifactRoleDeleteFile):
					errnie.Debug("fs.Store.buffer", "op", "delete", "path", path)
					if err = conn.Remove(path); err != nil {
						return errnie.Error(err)
					}
				case uint32(datura.ArtifactRoleListFiles):
					errnie.Debug("fs.Store.buffer", "op", "list", "path", path)
					if files, err = conn.Ls(path); err != nil {
						return errnie.Error(err)
					}
				}

				if fh != nil {
					defer conn.Close(fh)

					buf := bytes.NewBuffer([]byte{})
					if n, err = io.Copy(buf, fh); err != nil {
						return errnie.Error(err)
					}

					errnie.Debug("fs.Store.buffer", "n", n)

					datura.WithPayload(buf.Bytes())(artifact)
					return nil
				}

				if len(files) > 0 {
					buf := bytes.NewBuffer([]byte{})

					for _, file := range files {
						buf.WriteString(file + "\n")
					}

					if _, err = io.Copy(artifact, buf); err != nil {
						return errnie.Error(err)
					}
				}

				return nil
			}),
			conn: conn,
		}
	})

	return store
}

/*
Initialize loads embedded files into the store's filesystem from the given root directory.
*/
func (store *Store) Initialize(embedded embed.FS, root string) (err error) {
	return store.conn.Load(embedded, root)
}

/*
Read implements the io.Reader interface for the Store.
It reads data from the underlying buffer.
*/
func (s *Store) Read(p []byte) (n int, err error) {
	errnie.Debug("fs.Store.Read")
	return s.buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Store.
It writes data to the underlying buffer.
*/
func (s *Store) Write(p []byte) (n int, err error) {
	errnie.Debug("fs.Store.Write")
	return s.buffer.Write(p)
}

/*
Close implements the io.Closer interface for the Store.
It closes the underlying buffer.
*/
func (s *Store) Close() error {
	errnie.Debug("fs.Store.Close")
	return s.buffer.Close()
}
