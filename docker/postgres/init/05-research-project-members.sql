-- Project membership: Clerk user ids scoped to a research project.
CREATE TABLE IF NOT EXISTS research_project_members (
  research_project_id UUID NOT NULL REFERENCES research_projects (id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'member'
    CHECK (role IN ('owner', 'member')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (research_project_id, user_id)
);

ALTER TABLE research_project_members REPLICA IDENTITY FULL;

CREATE INDEX IF NOT EXISTS research_project_members_user_idx
  ON research_project_members (user_id);
