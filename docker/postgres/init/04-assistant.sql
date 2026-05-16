-- Assistant team: personas, sessions, messages.
-- All tables use REPLICA IDENTITY FULL so Electric can replicate row changes.
-- Scope semantics:
--   global   -> visible to everyone, owner_id NULL, organization_slug NULL
--   team     -> visible to one org, owner_id NULL, organization_slug NOT NULL
--   personal -> visible to one user, owner_id NOT NULL, organization_slug NULL

CREATE TABLE IF NOT EXISTS assistant_personas (
  id                UUID PRIMARY KEY,
  scope             TEXT NOT NULL CHECK (scope IN ('global','team','personal')),
  owner_id          TEXT,
  organization_slug TEXT,
  name              TEXT NOT NULL,
  system_prompt     TEXT NOT NULL DEFAULT '',
  model             TEXT NOT NULL,
  temperature       DOUBLE PRECISION NOT NULL DEFAULT 0.7,
  max_tokens        INTEGER NOT NULL DEFAULT 2048,
  adapter_type      TEXT NOT NULL DEFAULT 'openai'
                    CHECK (adapter_type IN ('openai','ollama','openai-compat')),
  endpoint_url      TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (
    (scope = 'global'   AND owner_id IS NULL     AND organization_slug IS NULL)
    OR (scope = 'team'     AND owner_id IS NULL     AND organization_slug IS NOT NULL)
    OR (scope = 'personal' AND owner_id IS NOT NULL AND organization_slug IS NULL)
  )
);
ALTER TABLE assistant_personas REPLICA IDENTITY FULL;

CREATE INDEX IF NOT EXISTS assistant_personas_scope_idx ON assistant_personas (scope);
CREATE INDEX IF NOT EXISTS assistant_personas_org_idx   ON assistant_personas (organization_slug);
CREATE INDEX IF NOT EXISTS assistant_personas_owner_idx ON assistant_personas (owner_id);

CREATE TABLE IF NOT EXISTS assistant_sessions (
  id                UUID PRIMARY KEY,
  scope             TEXT NOT NULL CHECK (scope IN ('team','personal')),
  owner_id          TEXT,
  organization_slug TEXT,
  title             TEXT NOT NULL DEFAULT 'New conversation',
  window_size       INTEGER NOT NULL DEFAULT 20,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (
       (scope = 'team'     AND owner_id IS NULL     AND organization_slug IS NOT NULL)
    OR (scope = 'personal' AND owner_id IS NOT NULL AND organization_slug IS NULL)
  )
);
ALTER TABLE assistant_sessions REPLICA IDENTITY FULL;

CREATE INDEX IF NOT EXISTS assistant_sessions_org_idx   ON assistant_sessions (organization_slug);
CREATE INDEX IF NOT EXISTS assistant_sessions_owner_idx ON assistant_sessions (owner_id);

CREATE TABLE IF NOT EXISTS assistant_session_personas (
  session_id UUID NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
  persona_id UUID NOT NULL REFERENCES assistant_personas(id) ON DELETE CASCADE,
  position   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (session_id, persona_id)
);
ALTER TABLE assistant_session_personas REPLICA IDENTITY FULL;

CREATE INDEX IF NOT EXISTS assistant_session_personas_session_idx ON assistant_session_personas (session_id);

CREATE TABLE IF NOT EXISTS assistant_messages (
  id           UUID PRIMARY KEY,
  session_id   UUID NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
  role         TEXT NOT NULL CHECK (role IN ('system','user','assistant')),
  parts        JSONB NOT NULL DEFAULT '[]'::jsonb,
  persona_id   UUID REFERENCES assistant_personas(id) ON DELETE SET NULL,
  persona_name TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE assistant_messages REPLICA IDENTITY FULL;

CREATE INDEX IF NOT EXISTS assistant_messages_session_idx ON assistant_messages (session_id, created_at);

-- Seed a default global persona so first-time users have something usable.
INSERT INTO assistant_personas (id, scope, name, system_prompt, model, temperature, max_tokens, adapter_type)
VALUES (
  '00000000-0000-0000-0000-000000000001',
  'global',
  'Assistant',
  'You are a helpful research assistant. You can search arXiv for papers when relevant.',
  'gpt-5.4-mini',
  0.7,
  2048,
  'openai'
)
ON CONFLICT (id) DO NOTHING;
