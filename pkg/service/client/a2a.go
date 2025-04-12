package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// A2AClient provides methods for communicating with an A2A server
type A2AClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewA2AClient creates a new A2A client to communicate with an A2A server
func NewA2AClient(baseURL string) *A2AClient {
	return &A2AClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// SendTask sends a task to the A2A server
func (c *A2AClient) SendTask(taskID string, message task.Message) (*task.Task, error) {
	// Create parameters
	params := map[string]interface{}{
		"taskId":  taskID,
		"message": message,
	}

	// Marshal params to JSON to create RawMessage
	paramsJSON, err := types.SimdMarshalJSON(params)

	if err != nil {
		return nil, fmt.Errorf("error marshaling params: %w", err)
	}

	// Create the JSON-RPC request
	request := types.JSONRPC{
		Version: "2.0",
		Method:  "tasks/send",
		ID:      taskID,
		Params:  paramsJSON,
	}

	// Marshal the request to JSON
	requestBody, err := types.SimdMarshalJSON(request)

	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	// Create and send the HTTP request
	resp, err := c.httpClient.Post(
		c.baseURL+"/rpc",
		"application/json",
		bytes.NewBuffer(requestBody),
	)

	if err != nil {
		return nil, fmt.Errorf("error sending request: %w", err)
	}

	defer resp.Body.Close()

	// Check the response status
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	// Decode the response
	var response types.JSONRPCResponse

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	// Check for errors in the response
	if response.Error != nil {
		return nil, fmt.Errorf("server returned error: %s", response.Error.Message)
	}

	// Convert the result to a Task
	resultJSON, err := types.SimdMarshalJSON(response.Result)

	if err != nil {
		return nil, fmt.Errorf("error marshaling result: %w", err)
	}

	var resultTask task.Task

	if err := types.SimdUnmarshalJSON(resultJSON, &resultTask); err != nil {
		return nil, fmt.Errorf("error unmarshaling result: %w", err)
	}

	return &resultTask, nil
}

// GetTask retrieves a task from the A2A server
func (c *A2AClient) GetTask(taskID string) (*task.Task, error) {
	// Create parameters
	params := map[string]interface{}{
		"id": taskID,
	}

	// Marshal params to JSON to create RawMessage
	paramsJSON, err := types.SimdMarshalJSON(params)
	if err != nil {
		return nil, fmt.Errorf("error marshaling params: %w", err)
	}

	// Create the JSON-RPC request
	request := types.JSONRPC{
		Version: "2.0",
		Method:  "tasks/get",
		ID:      taskID,
		Params:  paramsJSON,
	}

	// Marshal the request to JSON
	requestBody, err := types.SimdMarshalJSON(request)

	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	// Create and send the HTTP request
	resp, err := c.httpClient.Post(
		c.baseURL+"/rpc",
		"application/json",
		bytes.NewBuffer(requestBody),
	)

	if err != nil {
		return nil, fmt.Errorf("error sending request: %w", err)
	}

	defer resp.Body.Close()

	// Check the response status
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	// Decode the response
	var response types.JSONRPCResponse

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	// Check for errors in the response
	if response.Error != nil {
		return nil, fmt.Errorf("server returned error: %s", response.Error.Message)
	}

	// Convert the result to a Task
	resultJSON, err := types.SimdMarshalJSON(response.Result)

	if err != nil {
		return nil, fmt.Errorf("error marshaling result: %w", err)
	}

	var resultTask task.Task

	if err := types.SimdUnmarshalJSON(resultJSON, &resultTask); err != nil {
		return nil, fmt.Errorf("error unmarshaling result: %w", err)
	}

	return &resultTask, nil
}

// CancelTask cancels a task on the A2A server
func (c *A2AClient) CancelTask(taskID string) error {
	// Create parameters
	params := map[string]interface{}{
		"id": taskID,
	}

	// Marshal params to JSON to create RawMessage
	paramsJSON, err := types.SimdMarshalJSON(params)
	if err != nil {
		return fmt.Errorf("error marshaling params: %w", err)
	}

	// Create the JSON-RPC request
	request := types.JSONRPC{
		Version: "2.0",
		Method:  "tasks/cancel",
		ID:      taskID,
		Params:  paramsJSON,
	}

	// Marshal the request to JSON
	requestBody, err := types.SimdMarshalJSON(request)

	if err != nil {
		return fmt.Errorf("error marshaling request: %w", err)
	}

	// Create and send the HTTP request
	resp, err := c.httpClient.Post(
		c.baseURL+"/rpc",
		"application/json",
		bytes.NewBuffer(requestBody),
	)

	if err != nil {
		return fmt.Errorf("error sending request: %w", err)
	}

	defer resp.Body.Close()

	// Check the response status
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	// Decode the response
	var response types.JSONRPCResponse

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return fmt.Errorf("error decoding response: %w", err)
	}

	// Check for errors in the response
	if response.Error != nil {
		return fmt.Errorf("server returned error: %s", response.Error.Message)
	}

	return nil
}
