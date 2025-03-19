package fs

import (
	"embed"
	"io/fs"
	"os"

	"github.com/spf13/afero"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Conn represents a connection to an in-memory filesystem using afero.
*/
type Conn struct {
	memfs afero.Fs
}

/*
NewConn creates a new connection to an in-memory filesystem.
*/
func NewConn() *Conn {
	return &Conn{
		memfs: afero.NewMemMapFs(),
	}
}

/*
Load copies files from an embedded filesystem into the in-memory filesystem.
It walks through the embedded filesystem starting from the given root path.
*/
func (conn *Conn) Load(embedded embed.FS, root string) (err error) {
	return fs.WalkDir(embedded, root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return errnie.Error(err)
		}

		if d.IsDir() {
			return conn.memfs.MkdirAll(path, 0755)
		}

		conn.memfs.Create(path)
		return errnie.Error(err)
	})
}

/*
Ls lists all files in the given directory path.
Returns a slice of filenames and any error encountered.
*/
func (conn *Conn) Ls(path string) (files []string, err error) {
	dir, err := afero.ReadDir(conn.memfs, path)
	if err != nil {
		return nil, errnie.Error(err)
	}

	for _, file := range dir {
		files = append(files, file.Name())
	}

	return files, nil
}

/*
Open opens a file at the given path with read-write permissions.
Returns the file handle and any error encountered.
*/
func (conn *Conn) Open(path string) (file afero.File, err error) {
	return conn.memfs.OpenFile(path, os.O_RDWR, 0644)
}

/*
Save writes the given data to a file at the specified path.
*/
func (conn *Conn) Save(path string, data []byte) (err error) {
	return afero.WriteFile(conn.memfs, path, data, 0644)
}

/*
Remove deletes the file at the given path.
*/
func (conn *Conn) Remove(path string) (err error) {
	return conn.memfs.Remove(path)
}

/*
Close closes the given file handle.
*/
func (conn *Conn) Close(fh afero.File) (err error) {
	return fh.Close()
}
