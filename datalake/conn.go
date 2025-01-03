package datalake

import (
	"bytes"
	"context"
	"os"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/theapemachine/errnie"
)

/*
Conn is a wrapper around a datalake connection, which
uses S3 compatible storage.
*/
type Conn struct {
	client *minio.Client
	bucket string
	err    error
}

func NewConn() *Conn {
	endpoint := "localhost:9000"
	accessKeyID := os.Getenv("MINIO_USER")
	secretAccessKey := os.Getenv("MINIO_PASSWORD")
	useSSL := false

	client, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(accessKeyID, secretAccessKey, ""),
		Secure: useSSL,
	})

	if err != nil {
		errnie.Error(err)
		return nil
	}

	client.MakeBucket(
		context.Background(),
		"datalake",
		minio.MakeBucketOptions{Region: "us-east2"},
	)

	return &Conn{client: client, bucket: "datalake"}
}

func (conn *Conn) List(ctx context.Context, path string) <-chan minio.ObjectInfo {
	return conn.client.ListObjects(ctx, conn.bucket, minio.ListObjectsOptions{
		Prefix:       path,
		Recursive:    true,
		WithMetadata: true,
	})
}

func (conn *Conn) Get(ctx context.Context, path string) (*minio.Object, error) {
	var obj *minio.Object

	errnie.Info("attempting to get object from path: %s", path)
	
	if obj, conn.err = conn.client.GetObject(ctx, conn.bucket, path, minio.GetObjectOptions{}); conn.err != nil {
		errnie.Info("failed to get object: %v", conn.err)
		return nil, conn.err
	}

	// Check if the object exists by trying to get its stat
	_, err := obj.Stat()
	if err != nil {
		errnie.Info("object does not exist or cannot be accessed: %v", err)
		return nil, err
	}

	errnie.Info("successfully retrieved object from path: %s", path)
	return obj, nil
}

func (conn *Conn) Put(ctx context.Context, path string, data []byte, metadata map[string]string) (err error) {
	reader := bytes.NewReader(data)

	if _, err = conn.client.PutObject(
		ctx,
		conn.bucket,
		path,
		reader,
		int64(reader.Len()),
		minio.PutObjectOptions{
			UserMetadata: metadata,
		},
	); err != nil {
		return errnie.Error(err)
	}

	return err
}
