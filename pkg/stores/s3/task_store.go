package s3

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// S3TaskStore implements the task.TaskStore interface using S3.
type S3TaskStore struct {
	repo *Repository
}

// NewS3TaskStore creates a new S3-based task store.
func NewS3TaskStore(repo *Repository) *S3TaskStore {
	return &S3TaskStore{repo: repo}
}

// objectKey generates the S3 object key for a given task ID.
// We can adjust the prefix/path as needed.
func (s *S3TaskStore) objectKey(taskID string) string {
	return fmt.Sprintf("tasks/%s.json", taskID)
}

// CreateTask saves a new task to S3.
func (s *S3TaskStore) CreateTask(t *task.Task) error {
	return s.UpdateTask(t) // For S3, create and update are the same: PutObject
}

// mapS3Error maps S3-specific errors to appropriate task store errors
func mapS3Error(taskID string, err error) error {
	var noKey *types.NoSuchKey
	if errors.As(err, &noKey) {
		return task.NewTaskNotFoundError(taskID, err)
	}

	var noBucket *types.NoSuchBucket
	if errors.As(err, &noBucket) {
		return task.NewInternalError("GetTask", "S3 bucket not found", err, -32603)
	}

	// For other S3 errors, return a generic internal error
	return task.NewInternalError("GetTask", "S3 operation failed", err, -32603)
}

// GetTask retrieves a task from S3 by its ID.
func (s *S3TaskStore) GetTask(taskID string) (*task.Task, error) {
	key := s.objectKey(taskID)
	reader, err := s.repo.Get(key)
	if err != nil {
		return nil, mapS3Error(taskID, err)
	}
	defer reader.Close()

	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, task.NewInternalError("GetTask", "Failed to read task data", err, -32603)
	}

	t := &task.Task{}
	if err := json.Unmarshal(data, t); err != nil {
		return nil, task.NewInternalError("GetTask", "Failed to unmarshal task data", err, -32603)
	}

	return t, nil
}

// UpdateTask updates an existing task in S3 by overwriting it.
func (s *S3TaskStore) UpdateTask(t *task.Task) error {
	key := s.objectKey(t.ID)
	data, err := json.Marshal(t)
	if err != nil {
		return fmt.Errorf("failed to marshal task '%s' for S3: %w", t.ID, err)
	}

	body := bytes.NewReader(data)
	err = s.repo.Put(key, body)
	if err != nil {
		return fmt.Errorf("failed to put task '%s' to S3: %w", t.ID, err)
	}

	return nil
}
