-- Table used by the frontend Electric shape (table=research_projects) and insert API.
-- REPLICA IDENTITY FULL is required so Electric can replicate row changes correctly.
CREATE TABLE IF NOT EXISTS research_projects (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  organization_slug TEXT NOT NULL DEFAULT '',
  project_slug TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE research_projects REPLICA IDENTITY FULL;
