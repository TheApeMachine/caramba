package devteam

import (
	"fmt"

	"github.com/google/go-github/v67/github"
	"github.com/theapemachine/errnie"
)

func (orchestrator *Orchestrator) moveCard(cardID, columnKey, note string) error {
	query := `UPDATE kanban_cards SET column_key = $1, updated_at = NOW() WHERE id = $2`

	if _, err := orchestrator.db.ExecContext(orchestrator.ctx, query, columnKey, cardID); err != nil {
		return errnie.Error(
			fmt.Errorf("orchestrator: move card %s to %s: %w", cardID, columnKey, err),
			"card_id", cardID,
			"column_key", columnKey,
		)
	}

	if note == "" {
		return nil
	}

	desc := fmt.Sprintf("[devteam] %s", orchestrator.redact(note))
	if _, err := orchestrator.db.ExecContext(
		orchestrator.ctx,
		`UPDATE kanban_cards SET description = description || E'\n' || $1 WHERE id = $2`,
		desc, cardID,
	); err != nil {
		return errnie.Error(
			fmt.Errorf("orchestrator: append card %s move note: %w", cardID, err),
			"card_id", cardID,
			"column_key", columnKey,
		)
	}

	return nil
}

func (orchestrator *Orchestrator) openPR(event ColumnEvent, branch string) (string, error) {
	base := "main"
	title := event.Title
	body := fmt.Sprintf(
		"Automated PR from the Caramba AI dev team.\n\nFeature request card: %s\n\n%s",
		event.ID, event.Description,
	)

	pr, _, err := orchestrator.github.PullRequests.Create(
		orchestrator.ctx,
		orchestrator.cfg.GitHubOwner,
		orchestrator.cfg.GitHubRepo,
		&github.NewPullRequest{
			Title: &title,
			Head:  &branch,
			Base:  &base,
			Body:  &body,
		},
	)

	if err != nil {
		return "", fmt.Errorf("orchestrator: open PR: %w", err)
	}

	return pr.GetHTMLURL(), nil
}
