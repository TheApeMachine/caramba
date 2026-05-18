-- Research papers: synced via Electric (table=research_papers), scoped by organization_slug.
-- document: JSON with { metadata, blocks } matching the in-app paper editor payload.
-- revision: incremented on each successful save; clients append-only history in paper_revision_events.

CREATE TABLE IF NOT EXISTS research_papers (
  id UUID PRIMARY KEY,
  research_project_id UUID NOT NULL REFERENCES research_projects (id) ON DELETE CASCADE,
  organization_slug TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL DEFAULT '',
  document JSONB NOT NULL DEFAULT '{}'::jsonb,
  revision BIGINT NOT NULL DEFAULT 1,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS research_papers_organization_slug_idx
  ON research_papers (organization_slug);

CREATE INDEX IF NOT EXISTS research_papers_project_id_idx
  ON research_papers (research_project_id);

ALTER TABLE research_papers REPLICA IDENTITY FULL;

-- Append-only log for audit and collaborative deltas (patch optional: RFC 6902 JSON Patch).
CREATE TABLE IF NOT EXISTS paper_revision_events (
  id UUID PRIMARY KEY,
  paper_id UUID NOT NULL REFERENCES research_papers (id) ON DELETE CASCADE,
  revision BIGINT NOT NULL,
  previous_revision BIGINT NOT NULL,
  document JSONB NOT NULL,
  patch JSONB,
  summary TEXT NOT NULL DEFAULT '',
  actor_id TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS paper_revision_events_paper_revision_uidx
  ON paper_revision_events (paper_id, revision);

CREATE INDEX IF NOT EXISTS paper_revision_events_paper_created_idx
  ON paper_revision_events (paper_id, created_at);
