package fs

import (
	"bytes"
	"context"
	"embed"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/spf13/afero"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var once sync.Once
var store *Store

/*
Store represents a file system store with an in-memory filesystem and buffered operations.
*/
type Store struct {
	ctx    context.Context
	cancel context.CancelFunc
	conn   *Conn
}

/*
NewStore creates a new file system store with an in-memory filesystem.
It uses a singleton pattern to ensure only one store instance exists.
*/
func NewStore() *Store {
	errnie.Debug("fs.Store.New")
	conn := NewConn()

	ctx, cancel := context.WithCancel(context.Background())

	once.Do(func() {
		store = &Store{
			ctx:    ctx,
			cancel: cancel,
			conn:   conn,
		}
	})

	return store
}

/*
Generate processes file operations and returns artifacts with the results.
It implements the Generator pattern to handle file system operations asynchronously.
*/
func (s *Store) Generate(
	buffer chan *datura.ArtifactBuilder,
	fn ...func(artifact *datura.ArtifactBuilder) *datura.ArtifactBuilder,
) chan *datura.ArtifactBuilder {
	errnie.Debug("fs.Store.Generate")

	out := make(chan *datura.ArtifactBuilder)

	go func() {
		defer close(out)

		select {
		case <-s.ctx.Done():
			errnie.Debug("fs.Store.Generate.ctx.Done")
			s.cancel()
			return
		case artifact := <-buffer:
			// Extract file operation information from the artifact
			var (
				fh    afero.File
				files []string
				path  = datura.GetMetaValue[string](artifact, "path")
				n     int64
				err   error
			)

			// Process the artifact based on its role
			switch artifact.Role() {
			case uint32(datura.ArtifactRoleOpenFile):
				errnie.Debug("fs.Store.Generate", "op", "open", "path", path)
				if fh, err = s.conn.Open(path); err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}
			case uint32(datura.ArtifactRoleSaveFile):
				errnie.Debug("fs.Store.Generate", "op", "save", "path", path)
				if fh, err = s.conn.Open(path); err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}
			case uint32(datura.ArtifactRoleDeleteFile):
				errnie.Debug("fs.Store.Generate", "op", "delete", "path", path)
				if err = s.conn.Remove(path); err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}
				out <- datura.New(datura.WithEncryptedPayload([]byte("File deleted successfully")))
				return
			case uint32(datura.ArtifactRoleListFiles):
				errnie.Debug("fs.Store.Generate", "op", "list", "path", path)
				if files, err = s.conn.Ls(path); err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}
			default:
				out <- datura.New(datura.WithError(errnie.Error(fmt.Errorf("unknown file operation: %d", artifact.Role()))))
				return
			}

			// Handle file content if we have an open file handle
			if fh != nil {
				defer s.conn.Close(fh)

				buf := bytes.NewBuffer([]byte{})
				if n, err = io.Copy(buf, fh); err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}

				errnie.Debug("fs.Store.Generate", "n", n)
				out <- datura.New(datura.WithEncryptedPayload(buf.Bytes()))
				return
			}

			// Handle file listing
			if len(files) > 0 {
				buf := bytes.NewBuffer([]byte{})
				for _, file := range files {
					buf.WriteString(file + "\n")
				}
				out <- datura.New(datura.WithEncryptedPayload(buf.Bytes()))
				return
			}

			// If no operation was performed
			out <- datura.New(datura.WithError(errnie.Error(fmt.Errorf("no valid file operation performed"))))
		}
	}()

	return out
}

/*
Initialize loads embedded files into the store's filesystem from the given root directory.
*/
func (store *Store) Initialize(embedded embed.FS, root string) (err error) {
	return store.conn.Load(embedded, root)
}

/*
Stat implements the afero.Stat interface for the Store.
It returns the file info for the given path.
*/
func (store *Store) Stat(path string) (fi os.FileInfo, err error) {
	return store.conn.memfs.Stat(path)
}
