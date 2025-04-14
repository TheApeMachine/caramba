package task

import "github.com/theapemachine/caramba/pkg/auth"

// PushNotificationConfig represents configuration for push notifications
type PushNotificationConfig struct {
	URL            string               `json:"url"`
	Token          *string              `json:"token,omitempty"`
	Authentication *auth.Authentication `json:"authentication,omitempty"`
}

// TaskPushNotificationConfig represents push notification configuration for a task
type TaskPushNotificationConfig struct {
	ID                     string                 `json:"id"`
	PushNotificationConfig PushNotificationConfig `json:"pushNotificationConfig"`
}
