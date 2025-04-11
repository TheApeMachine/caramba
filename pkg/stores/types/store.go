package stores

import "io"

type Store interface {
	Get(objectKey string) (io.ReadCloser, error)
	Put(objectKey string, body io.Reader) error
}
