package task

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/theapemachine/caramba/pkg/auth"
)

type PushNotificationConfig struct {
	URL            string              `json:"url"`
	Token          string              `json:"token"` // token unique to this task/session
	Authentication auth.Authentication `json:"authentication"`
}

type TaskPushNotificationConfig struct {
	ID               string                 `json:"id"` //task id
	PushNotification PushNotificationConfig `json:"pushNotification"`
}

// PushNotification represents a notification to be sent to a client
type PushNotification struct {
	TaskID   string `json:"taskId"`
	Type     string `json:"type"`
	Payload  any    `json:"payload"`
	Metadata any    `json:"metadata,omitempty"`
}

// NotificationManager handles sending notifications
type NotificationManager struct {
	httpClient *http.Client
	configs    map[string]PushNotificationConfig
}

// NewNotificationManager creates a new notification manager
func NewNotificationManager() *NotificationManager {
	return &NotificationManager{
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
		configs: make(map[string]PushNotificationConfig),
	}
}

// RegisterPushConfig registers a push configuration for a task
func (nm *NotificationManager) RegisterPushConfig(taskID string, config PushNotificationConfig) {
	nm.configs[taskID] = config
	log.Printf("Registered push notification config for task %s to URL: %s", taskID, config.URL)
}

// RemovePushConfig removes a push configuration for a task
func (nm *NotificationManager) RemovePushConfig(taskID string) {
	delete(nm.configs, taskID)
}

// SendTaskStatusUpdate sends a task status update notification
func (nm *NotificationManager) SendTaskStatusUpdate(
	taskID string,
	status TaskStatus,
	final bool,
	metadata map[string]any,
) error {
	config, exists := nm.configs[taskID]
	if !exists {
		// No configured push endpoint for this task
		return nil
	}

	notification := PushNotification{
		TaskID: taskID,
		Type:   "status_update",
		Payload: TaskStatusUpdateEvent{
			ID:       taskID,
			Status:   status,
			Final:    final,
			Metadata: metadata,
		},
	}

	return nm.sendNotification(config, notification)
}

// SendTaskArtifactUpdate sends a task artifact update notification
func (nm *NotificationManager) SendTaskArtifactUpdate(taskID string, artifact Artifact, metadata map[string]any) error {
	config, exists := nm.configs[taskID]
	if !exists {
		// No configured push endpoint for this task
		return nil
	}

	notification := PushNotification{
		TaskID: taskID,
		Type:   "artifact_update",
		Payload: TaskArtifactUpdateEvent{
			ID:       taskID,
			Artifact: artifact,
			Metadata: metadata,
		},
	}

	return nm.sendNotification(config, notification)
}

// sendNotification sends a notification to the configured webhook URL
func (nm *NotificationManager) sendNotification(config PushNotificationConfig, notification PushNotification) error {
	payload, err := json.Marshal(notification)
	if err != nil {
		return fmt.Errorf("error marshaling notification: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, config.URL, bytes.NewBuffer(payload))
	if err != nil {
		return fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// Add authentication if token is provided
	if config.Token != "" {
		if config.Authentication.Schemes == "bearer" {
			req.Header.Set("Authorization", "Bearer "+config.Token)
		} else {
			req.Header.Set("X-API-Key", config.Token)
		}
	}

	resp, err := nm.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("error sending notification: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("error response from notification endpoint: %d", resp.StatusCode)
	}

	log.Printf("Successfully sent notification to %s for task %s", config.URL, notification.TaskID)
	return nil
}
