-- A team chooses how its data is stored. 'shared' (default) means the team and
-- everything scoped to it (projects, cards, papers) sync through Electric and
-- live in this server's Postgres. 'local' means the team is single-device:
-- nothing about the team or its children ever leaves the creator's browser.
ALTER TABLE teams
  ADD COLUMN IF NOT EXISTS privacy_mode TEXT NOT NULL DEFAULT 'shared'
    CHECK (privacy_mode IN ('shared', 'local'));

-- A team carries a small bit of presentation metadata so the dashboard and
-- switcher can distinguish teams at a glance without piling more tables on.
ALTER TABLE teams
  ADD COLUMN IF NOT EXISTS color TEXT NOT NULL DEFAULT '';
ALTER TABLE teams
  ADD COLUMN IF NOT EXISTS emoji TEXT NOT NULL DEFAULT '';
