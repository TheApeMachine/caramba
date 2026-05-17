package devteam

import (
	"errors"
	"fmt"
)

func (orchestrator *Orchestrator) runPlanner(
	event ColumnEvent, blastContext string,
) ([]SubtaskDraft, error) {
	agentID := "planner-" + event.ID[:8]

	sandbox, err := NewSandbox(orchestrator.ctx, SandboxConfig{
		AgentID:       agentID,
		Image:         orchestrator.cfg.DockerImage,
		GitHubToken:   orchestrator.cfg.GitHubToken,
		GitHubOwner:   orchestrator.cfg.GitHubOwner,
		GitHubRepo:    orchestrator.cfg.GitHubRepo,
		FeatureBranch: featureBranch(event),
	})

	if err != nil {
		return nil, fmt.Errorf("planner sandbox: %w", err)
	}

	defer sandbox.Destroy()

	if err := sandbox.Start(); err != nil {
		return nil, fmt.Errorf("planner sandbox start: %w", err)
	}

	editor := NewVirtualEditor(agentID, sandbox, orchestrator.locks)
	planner := NewPlanner(orchestrator.ctx, orchestrator.cfg.Planner, editor)

	result, err := planner.Plan(event.Title, event.Description, blastContext)

	if err != nil {
		return nil, err
	}

	return result.Subtasks, nil
}

/*
persistSubtasks writes the Planner's drafts to the database, annotating each
subtask's context snapshot with sibling awareness so developers know what
neighbouring agents are working on.
*/
func (orchestrator *Orchestrator) persistSubtasks(
	event ColumnEvent,
	drafts []SubtaskDraft,
	blastContext string,
) ([]string, error) {
	ids := make([]string, 0, len(drafts))

	for index, draft := range drafts {
		snap := SubtaskContext{
			BlastRadius:  blastContext,
			KeySymbols:   draft.KeySymbols,
			FilesInScope: draft.FilesInScope,
			SiblingNotes: draft.SiblingNotes,
		}

		id, err := orchestrator.subtasks.Insert(
			orchestrator.ctx,
			event.ID,
			index,
			draft.Title,
			draft.Description,
			snap,
		)

		if err != nil {
			return nil, err
		}

		ids = append(ids, id)
	}

	return ids, nil
}

/*
runSubtask executes the full developer-reviewer loop for a single subtask in
its own ephemeral sandbox on the shared feature branch.
*/
func (orchestrator *Orchestrator) runSubtask(
	event ColumnEvent, subtask Subtask, branch string,
) (runErr error) {
	agentID := "dev-" + subtask.ID[:8]

	defer orchestrator.locks.ReleaseAll(agentID)

	if err := orchestrator.setSubtaskStatus(subtask.ID, "in-progress", agentID); err != nil {
		return err
	}

	defer func() {
		if runErr == nil {
			return
		}

		if err := orchestrator.setSubtaskStatus(subtask.ID, "failed", agentID); err != nil {
			runErr = errors.Join(runErr, err)
		}
	}()

	sandbox, err := NewSandbox(orchestrator.ctx, SandboxConfig{
		AgentID:       agentID,
		Image:         orchestrator.cfg.DockerImage,
		GitHubToken:   orchestrator.cfg.GitHubToken,
		GitHubOwner:   orchestrator.cfg.GitHubOwner,
		GitHubRepo:    orchestrator.cfg.GitHubRepo,
		FeatureBranch: branch,
	})

	if err != nil {
		return fmt.Errorf("sandbox: %w", err)
	}

	defer sandbox.Destroy()

	if err := sandbox.Start(); err != nil {
		return fmt.Errorf("sandbox start: %w", err)
	}

	editor := NewVirtualEditor(agentID, sandbox, orchestrator.locks)
	developer := NewDeveloper(orchestrator.ctx, orchestrator.cfg.Developer, editor)
	reviewer := NewReviewer(orchestrator.ctx, orchestrator.cfg.Reviewer)
	subtaskContext := formatSubtaskContext(subtask)

	feedback := ""
	passedReview := false
	verdict := ReviewVerdict{}

	for range reviewerMaxIterations {
		if err := developer.Implement(
			subtask.Title, subtask.Description, subtaskContext, feedback,
		); err != nil {
			return err
		}

		verdict, err = reviewer.Review(sandbox, subtask.Title, subtask.Description)

		if err != nil {
			return err
		}

		if verdict.Pass {
			passedReview = true
			break
		}

		feedback = verdict.Feedback
	}

	if !passedReview {
		return fmt.Errorf(
			"reviewer rejected after %d iterations: %s",
			reviewerMaxIterations,
			truncate(verdict.Feedback, 800),
		)
	}

	commitMsg := fmt.Sprintf("feat(%s): %s", event.ID[:8], subtask.Title)
	orchestrator.integrationMu.Lock()
	defer orchestrator.integrationMu.Unlock()

	if err := sandbox.CommitAndPush(commitMsg); err != nil {
		return fmt.Errorf("commit/push: %w", err)
	}

	if err := orchestrator.setSubtaskStatus(subtask.ID, "done", agentID); err != nil {
		return err
	}

	return nil
}

/*
finalise opens the pull request for the completed feature branch and moves the
parent card to the review column.
*/
func (orchestrator *Orchestrator) finalise(event ColumnEvent, branch string) error {
	prURL, err := orchestrator.openPR(event, branch)

	if err != nil {
		return err
	}

	return orchestrator.moveCard(event.ID, "review", prURL)
}
