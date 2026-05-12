package devteam

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"
)

/*
Subtask is a single unit of work within a kanban card, created by the Planner
agent and executed by an independent Developer agent.
*/
type Subtask struct {
	ID              string
	CardID          string
	SortOrder       int
	Title           string
	Description     string
	Status          string
	AssignedAgent   string
	ContextSnapshot SubtaskContext
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

/*
SubtaskContext is the Planner's focused analysis for a single subtask: the
narrow blast radius, relevant symbols, and notes about what neighbouring
subtasks are doing so the developer can avoid conflicts.
*/
type SubtaskContext struct {
	BlastRadius  string            `json:"blast_radius"`
	KeySymbols   []string          `json:"key_symbols"`
	FilesInScope []string          `json:"files_in_scope"`
	SiblingNotes map[string]string `json:"sibling_notes"` // subtask title → note
}

/*
SubtaskStore handles all database operations for kanban_subtasks.
*/
type SubtaskStore struct {
	db *sql.DB
}

/*
NewSubtaskStore wraps an existing *sql.DB.
*/
func NewSubtaskStore(db *sql.DB) *SubtaskStore {
	return &SubtaskStore{db: db}
}

/*
Insert writes a new subtask row and returns the generated UUID.
*/
func (store *SubtaskStore) Insert(
	ctx context.Context, cardID string, sortOrder int, title, description string, snap SubtaskContext,
) (string, error) {
	snapJSON, err := json.Marshal(snap)

	if err != nil {
		return "", fmt.Errorf("subtask: marshal context: %w", err)
	}

	var id string

	err = store.db.QueryRowContext(ctx, `
		INSERT INTO kanban_subtasks
		  (card_id, sort_order, title, description, context_snapshot)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id
	`, cardID, sortOrder, title, description, string(snapJSON)).Scan(&id)

	if err != nil {
		return "", fmt.Errorf("subtask: insert: %w", err)
	}

	return id, nil
}

/*
ListForCard returns all subtasks for a card ordered by sort_order.
*/
func (store *SubtaskStore) ListForCard(ctx context.Context, cardID string) ([]Subtask, error) {
	rows, err := store.db.QueryContext(ctx, `
		SELECT id, card_id, sort_order, title, description,
		       status, COALESCE(assigned_agent, ''), context_snapshot,
		       created_at, updated_at
		FROM kanban_subtasks
		WHERE card_id = $1
		ORDER BY sort_order ASC
	`, cardID)

	if err != nil {
		return nil, fmt.Errorf("subtask: list: %w", err)
	}

	defer rows.Close()

	var subtasks []Subtask

	for rows.Next() {
		var st Subtask
		var snapRaw string

		if err := rows.Scan(
			&st.ID, &st.CardID, &st.SortOrder, &st.Title, &st.Description,
			&st.Status, &st.AssignedAgent, &snapRaw,
			&st.CreatedAt, &st.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("subtask: scan: %w", err)
		}

		if err := json.Unmarshal([]byte(snapRaw), &st.ContextSnapshot); err != nil {
			return nil, fmt.Errorf("subtask: unmarshal context: %w", err)
		}

		subtasks = append(subtasks, st)
	}

	return subtasks, rows.Err()
}

/*
SetStatus transitions a subtask to a new status and optionally records the
agent that owns it.
*/
func (store *SubtaskStore) SetStatus(
	ctx context.Context, subtaskID, status, agentID string,
) error {
	_, err := store.db.ExecContext(ctx, `
		UPDATE kanban_subtasks
		SET status = $1, assigned_agent = $2, updated_at = NOW()
		WHERE id = $3
	`, status, agentID, subtaskID)

	if err != nil {
		return fmt.Errorf("subtask: set status: %w", err)
	}

	return nil
}

/*
AllDone reports whether every subtask for cardID has status "done".
*/
func (store *SubtaskStore) AllDone(ctx context.Context, cardID string) (bool, error) {
	var pending int

	err := store.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM kanban_subtasks
		WHERE card_id = $1 AND status <> 'done'
	`, cardID).Scan(&pending)

	if err != nil {
		return false, fmt.Errorf("subtask: all-done check: %w", err)
	}

	return pending == 0, nil
}

/*
AnyFailed reports whether any subtask for cardID has status "failed".
*/
func (store *SubtaskStore) AnyFailed(ctx context.Context, cardID string) (bool, error) {
	var failed int

	err := store.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM kanban_subtasks
		WHERE card_id = $1 AND status = 'failed'
	`, cardID).Scan(&failed)

	if err != nil {
		return false, fmt.Errorf("subtask: any-failed check: %w", err)
	}

	return failed > 0, nil
}
