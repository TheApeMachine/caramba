-- Teams are the unit between a Clerk organization and a research project.
-- An org owns many teams; a user can be a member of many teams; a project belongs to one team.
CREATE TABLE IF NOT EXISTS teams (
  id UUID PRIMARY KEY,
  organization_slug TEXT NOT NULL,
  name TEXT NOT NULL,
  slug TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (organization_slug, slug)
);

ALTER TABLE teams REPLICA IDENTITY FULL;

CREATE INDEX IF NOT EXISTS teams_org_idx ON teams (organization_slug);

-- Membership of Clerk users in teams. Independent of project membership.
CREATE TABLE IF NOT EXISTS team_memberships (
  team_id UUID NOT NULL REFERENCES teams (id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'member'
    CHECK (role IN ('owner', 'member')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (team_id, user_id)
);

ALTER TABLE team_memberships REPLICA IDENTITY FULL;

CREATE INDEX IF NOT EXISTS team_memberships_user_idx ON team_memberships (user_id);

-- A research project belongs to at most one team. Left null on existing rows so
-- the migration is non-destructive; new projects created through the team-aware
-- flow will populate it.
ALTER TABLE research_projects
  ADD COLUMN IF NOT EXISTS team_id UUID REFERENCES teams (id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS research_projects_team_idx ON research_projects (team_id);
