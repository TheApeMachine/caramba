package handlers

import (
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/stores/s3"
	"github.com/theapemachine/caramba/pkg/task"
)

// taskArtifactGetParams defines the expected parameters for the task.artifact.get method
type taskArtifactGetParams struct {
	ArtifactID string `json:"artifact_id"` // Assuming A2A spec uses artifact_id
	// TaskID might also be relevant depending on how artifacts are organized
	// TaskID string `json:"task_id"`
}

// Modified signature to accept S3 repository and bucket name
func HandleTaskArtifactGet(store task.TaskStore, s3Repo *s3.Repository, bucketName string, params json.RawMessage) (interface{}, *task.TaskRequestError) {
	var getParams taskArtifactGetParams
	if err := types.SimdUnmarshalJSON(params, &getParams); err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.artifact.get",
			Data:    err.Error(),
		}
	}

	if getParams.ArtifactID == "" {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.artifact.get",
			Data:    "Missing required parameter: artifact_id",
		}
	}

	// Retrieve the artifact from S3
	artifactReadCloser, err := s3Repo.Get(getParams.ArtifactID)
	if err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32001, // Example: Application-specific error code for retrieval failure
			Message: "Artifact retrieval failed",
			Data:    err.Error(),
		}
	}
	defer artifactReadCloser.Close()

	// Read the artifact content
	artifactBytes, err := io.ReadAll(artifactReadCloser)
	if err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32002, // Example: Application-specific error code for read failure
			Message: "Failed to read artifact content",
			Data:    err.Error(),
		}
	}

	// Return artifact information including content
	// Note: Returning raw bytes might not be ideal. Consider base64 encoding or returning a URL.
	return map[string]interface{}{
		"artifact_id": getParams.ArtifactID,
		"status":      "retrieved",
		"content":     string(artifactBytes), // Returning content as string for simplicity now
	}, nil
}
