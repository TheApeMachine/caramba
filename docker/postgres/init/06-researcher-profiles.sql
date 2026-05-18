-- Per-researcher profile metadata (Clerk user id). Not Electric-synced; loaded via API.
CREATE TABLE IF NOT EXISTS researcher_profiles (
  user_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL DEFAULT '',
  role_title TEXT NOT NULL DEFAULT '',
  affiliation TEXT NOT NULL DEFAULT '',
  bio TEXT NOT NULL DEFAULT '',
  website TEXT NOT NULL DEFAULT '',
  research_focus TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS researcher_profiles_updated_idx
  ON researcher_profiles (updated_at DESC);
