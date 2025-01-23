package datalake

import (
	"bytes"
	"context"
	"fmt"
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

/*
NewConn creates a new connection to the S3-compatible storage.
It initializes the Minio client with the provided credentials and
creates a default bucket if it doesn't exist.

Returns:
  - *Conn: A new connection instance with initialized client and bucket
*/
func NewConn() *Conn {
	endpoint := "localhost:9000"
	accessKeyID := os.Getenv("MINIO_USER")
	secretAccessKey := os.Getenv("MINIO_PASSWORD")
	useSSL := false

	client, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(accessKeyID, secretAccessKey, ""),
		Secure: useSSL,
	})

	if err == nil {
		client.MakeBucket(
			context.Background(),
			"datalake",
			minio.MakeBucketOptions{Region: "us-east2"},
		)
	}

	return &Conn{err: err, client: client, bucket: "datalake"}
}

/*
Error implements the error interface.
*/
func (conn *Conn) Error() string {
	return conn.err.Error()
}

/*
IsConnected returns true if the connection is successful.
*/
func (conn *Conn) IsConnected() bool {
	healthy := conn.err == nil && conn.client != nil && conn.client.IsOnline()
	fmt.Println("conn.err", conn.err)
	fmt.Println("conn.client", conn.client)
	fmt.Println("conn.client.IsOnline()", conn.client.IsOnline())
	fmt.Println("healthy", healthy)
	return healthy
}

/*
List returns a channel of ObjectInfo for all objects in the specified path.
It recursively lists all objects and includes their metadata.

Parameters:
  - ctx: Context for the operation
  - path: The path prefix to list objects from

Returns:
  - <-chan minio.ObjectInfo: Channel streaming object information
*/
func (conn *Conn) List(ctx context.Context, path string) <-chan minio.ObjectInfo {
	return conn.client.ListObjects(ctx, conn.bucket, minio.ListObjectsOptions{
		Prefix:       path,
		Recursive:    true,
		WithMetadata: true,
	})
}

/*
Get retrieves an object from the specified path in the datalake.
It verifies the object's existence before returning it.

Parameters:
  - ctx: Context for the operation
  - path: The path to the object to retrieve

Returns:
  - *minio.Object: The retrieved object
  - error: Any error that occurred during retrieval
*/
func (conn *Conn) Get(ctx context.Context, path string) (*minio.Object, error) {
	var obj *minio.Object

	errnie.Info("attempting to get object from path: %s", path)

	if obj, conn.err = conn.client.GetObject(ctx, conn.bucket, path, minio.GetObjectOptions{}); conn.err != nil {
		errnie.Info("failed to get object: %v", conn.err)
		return nil, conn.err
	}

	// Check if the object exists by trying to get its stat
	_, conn.err = obj.Stat()
	if conn.err != nil {
		errnie.Info("object does not exist or cannot be accessed: %v", conn.err)
		return nil, conn.err
	}

	errnie.Info("successfully retrieved object from path: %s", path)
	return obj, nil
}

/*
Put stores data at the specified path in the datalake with optional metadata.
It creates a new object or overwrites an existing one.

Parameters:
  - ctx: Context for the operation
  - path: The path where the object should be stored
  - data: The byte data to store
  - metadata: Optional metadata to attach to the object

Returns:
  - error: Any error that occurred during the operation
*/
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
