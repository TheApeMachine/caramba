package s3

import (
	"context"
	"io"
)

type Repository struct {
	conn       *Conn
	bucketName string
}

func NewRepository(conn *Conn, bucketName string) *Repository {
	return &Repository{conn: conn, bucketName: bucketName}
}

func (repo *Repository) Get(objectKey string) (io.ReadCloser, error) {
	return repo.conn.Get(context.Background(), repo.bucketName, objectKey)
}

func (repo *Repository) Put(objectKey string, body io.Reader) error {
	return repo.conn.Put(context.Background(), repo.bucketName, objectKey, body)
}
