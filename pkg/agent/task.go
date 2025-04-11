package agent

type Task struct {
	ID        string         `json:"id"`
	SessionID string         `json:"sessionId"`
	Status    TaskStatus     `json:"status"`
	History   []Message      `json:"history"`
	Artifacts []Artifact     `json:"artifacts"`
	Metadata  map[string]any `json:"metadata"`
}

type TaskStatus struct {
	State     TaskState `json:"state"`
	Message   Message   `json:"message"`
	Timestamp string    `json:"timestamp"`
}

type TaskStatusUpdateEvent struct {
	ID       string         `json:"id"`
	Status   TaskStatus     `json:"status"`
	Final    bool           `json:"final"`
	Metadata map[string]any `json:"metadata"`
}

type TaskArtifactUpdateEvent struct {
	ID       string         `json:"id"`
	Artifact Artifact       `json:"artifact"`
	Metadata map[string]any `json:"metadata"`
}

type TaskSendParams struct {
	ID               string                 `json:"id"`
	SessionID        string                 `json:"sessionId"`
	Message          Message                `json:"message"`
	HistoryLength    int                    `json:"historyLength"`
	PushNotification PushNotificationConfig `json:"pushNotification"`
	Metadata         map[string]any         `json:"metadata"`
}

type TaskState int

const (
	TaskStateSubmitted TaskState = iota
	TaskStateWorking
	TaskStateInputRequired
	TaskStateCompleted
	TaskStateCanceled
	TaskStateFailed
	TaskStateUnknown
)

func (state TaskState) String() string {
	return []string{
		"submitted",
		"working",
		"input-required",
		"completed",
		"canceled",
		"failed",
		"unknown",
	}[state]
}
