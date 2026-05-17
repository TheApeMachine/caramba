package devteam

import (
	"fmt"
	"time"
)

const subtaskStatusAttempts = 3

/*
setSubtaskStatus records a subtask transition with bounded retry so transient
database errors do not silently strand cards in an active column.
*/
func (orchestrator *Orchestrator) setSubtaskStatus(
	subtaskID, status, agentID string,
) error {
	var lastErr error

	for attempt := 1; attempt <= subtaskStatusAttempts; attempt++ {
		if err := orchestrator.subtasks.SetStatus(
			orchestrator.ctx,
			subtaskID,
			status,
			agentID,
		); err != nil {
			lastErr = err
			time.Sleep(time.Duration(attempt) * 200 * time.Millisecond)

			continue
		}

		return nil
	}

	return fmt.Errorf("orchestrator: set subtask %s status %s: %w", subtaskID, status, lastErr)
}

func (orchestrator *Orchestrator) redact(text string) string {
	if orchestrator == nil || orchestrator.cfg == nil {
		return text
	}

	return redactSecrets(text, orchestrator.cfg.GitHubToken)
}
