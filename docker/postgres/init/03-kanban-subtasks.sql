-- Subtask todo-list for kanban cards.
--
-- Each card can have N subtasks created by the Planner agent. They are rendered
-- as a todo-list on the card (not as separate board cards). The orchestrator
-- drives each subtask through its own developer → reviewer loop; the parent
-- card only advances to "review" once every subtask reaches "done".
--
-- context_snapshot stores the Planner's focused blast-radius JSON so the
-- implementing developer agent starts with pre-shaped context rather than
-- having to re-derive it from scratch.

CREATE TABLE IF NOT EXISTS kanban_subtasks (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  card_id          UUID NOT NULL REFERENCES kanban_cards (id) ON DELETE CASCADE,
  sort_order       INTEGER NOT NULL DEFAULT 0,
  title            TEXT NOT NULL,
  description      TEXT NOT NULL DEFAULT '',
  status           TEXT NOT NULL DEFAULT 'todo',
  assigned_agent   TEXT,
  context_snapshot JSONB NOT NULL DEFAULT '{}',
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT kanban_subtasks_status_chk CHECK (
    status IN ('todo', 'in-progress', 'done', 'failed')
  )
);

ALTER TABLE kanban_subtasks REPLICA IDENTITY FULL;

CREATE INDEX IF NOT EXISTS kanban_subtasks_card_idx    ON kanban_subtasks (card_id);
CREATE INDEX IF NOT EXISTS kanban_subtasks_status_idx  ON kanban_subtasks (card_id, status);

-- Notify the orchestrator when a subtask's status changes so the fan-out
-- scheduler can react without polling.
CREATE OR REPLACE FUNCTION kanban_subtasks_status_notify() RETURNS trigger AS $$
BEGIN
  IF NEW.status IS DISTINCT FROM OLD.status THEN
    PERFORM pg_notify(
      'kanban_subtask_change',
      json_build_object(
        'id',         NEW.id,
        'card_id',    NEW.card_id,
        'status',     NEW.status,
        'old_status', OLD.status
      )::text
    );
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS kanban_subtasks_status_change ON kanban_subtasks;

CREATE TRIGGER kanban_subtasks_status_change
  AFTER UPDATE ON kanban_subtasks
  FOR EACH ROW EXECUTE FUNCTION kanban_subtasks_status_notify();
