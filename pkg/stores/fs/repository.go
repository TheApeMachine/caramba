package fs

import (
	"context"
	"embed"
	"os"
	"sync"

	"github.com/spf13/afero"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var once sync.Once
var store *Store

/*
Store is an in-memory file system, used to work with the embedded filesystem.
This is primarily used by the Browser tool to load embedded scripts.
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
Get returns a file from the store.
*/
func (s *Store) Get(path string) (fh afero.File, err error) {
	return s.conn.Open(path)
}

/*
Put reads the content of a file handle and saves it to the store.
*/
func (s *Store) Put(path string, fh afero.File) (err error) {
	defer fh.Close() // Ensure file handle is closed
	content, err := afero.ReadAll(fh)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}
	return s.conn.Save(path, content) // Pass the content (byte slice)
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
