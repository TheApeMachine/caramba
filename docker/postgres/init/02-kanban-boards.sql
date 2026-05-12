-- Org / slug metadata for multi-board Kanban routing (Electric syncs research_projects).
ALTER TABLE research_projects ADD COLUMN IF NOT EXISTS organization_slug TEXT NOT NULL DEFAULT '';
ALTER TABLE research_projects ADD COLUMN IF NOT EXISTS project_slug TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS research_projects_organization_project_slug_uidx
  ON research_projects (organization_slug, project_slug)
  WHERE project_slug IS NOT NULL AND organization_slug <> '';

-- Canonical backlog intake board for inbound feature requests (Clerk org slug "caramba").
INSERT INTO research_projects (
  id,
  name,
  description,
  organization_slug,
  project_slug,
  created_at,
  updated_at
)
VALUES (
  'f47ac10b-58cc-4372-a567-0e02b2c3d479',
  'Requests',
  'Inbound feature requests from users.',
  'caramba',
  'requests',
  NOW(),
  NOW()
)
ON CONFLICT (id) DO NOTHING;

-- Kanban cards (Electric shape table=kanban_cards).
CREATE TABLE IF NOT EXISTS kanban_cards (
  id UUID PRIMARY KEY,
  research_project_id UUID NOT NULL REFERENCES research_projects (id) ON DELETE CASCADE,
  column_key TEXT NOT NULL,
  sort_order INTEGER NOT NULL DEFAULT 0,
  title TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  priority TEXT NOT NULL DEFAULT 'medium',
  labels_json TEXT NOT NULL DEFAULT '[]',
  assignees_json TEXT NOT NULL DEFAULT '[]',
  due_date TIMESTAMPTZ,
  requested_by TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT kanban_cards_column_key_chk CHECK (
    column_key IN ('backlog', 'todo', 'in-progress', 'review', 'done')
  ),
  CONSTRAINT kanban_cards_priority_chk CHECK (
    priority IN ('low', 'medium', 'high', 'critical')
  )
);

ALTER TABLE kanban_cards REPLICA IDENTITY FULL;

CREATE INDEX IF NOT EXISTS kanban_cards_project_idx ON kanban_cards (research_project_id);
CREATE INDEX IF NOT EXISTS kanban_cards_project_column_idx ON kanban_cards (research_project_id, column_key);

-- Notify the devteam orchestrator whenever a card's column_key changes.
-- Payload JSON: { "id": "...", "research_project_id": "...", "column_key": "...", "old_column_key": "..." }
CREATE OR REPLACE FUNCTION kanban_cards_column_notify() RETURNS trigger AS $$
BEGIN
  IF NEW.column_key IS DISTINCT FROM OLD.column_key THEN
    PERFORM pg_notify(
      'kanban_column_change',
      json_build_object(
        'id',                  NEW.id,
        'research_project_id', NEW.research_project_id,
        'column_key',          NEW.column_key,
        'old_column_key',      OLD.column_key,
        'title',               NEW.title,
        'description',         NEW.description
      )::text
    );
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS kanban_cards_column_change ON kanban_cards;

CREATE TRIGGER kanban_cards_column_change
  AFTER UPDATE ON kanban_cards
  FOR EACH ROW EXECUTE FUNCTION kanban_cards_column_notify();
