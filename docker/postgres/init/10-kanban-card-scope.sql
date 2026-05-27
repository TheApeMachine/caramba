-- Denormalize the parent project's organization_slug and team_id onto each
-- kanban card. Boards scope by project, team, or organization; kanban_cards
-- itself only carries research_project_id, so without these columns an org or
-- team board has to sync the whole table and filter client-side. With them,
-- each Electric shape becomes a simple indexed equality filter
-- (research_project_id = $1 / team_id = $1 / organization_slug = $1). This
-- mirrors the organization_slug denormalization already used on
-- research_paper_blocks so the shape proxy can filter without a join.

ALTER TABLE kanban_cards
  ADD COLUMN IF NOT EXISTS organization_slug TEXT NOT NULL DEFAULT '';
ALTER TABLE kanban_cards
  ADD COLUMN IF NOT EXISTS team_id UUID;

CREATE INDEX IF NOT EXISTS kanban_cards_org_idx  ON kanban_cards (organization_slug);
CREATE INDEX IF NOT EXISTS kanban_cards_team_idx ON kanban_cards (team_id);

-- Card side: on insert, or whenever a card is re-parented to a different
-- project, copy the parent project's scope onto the card. The FK on
-- research_project_id guarantees the SELECT finds exactly one project row.
CREATE OR REPLACE FUNCTION kanban_cards_fill_scope() RETURNS trigger AS $$
BEGIN
  SELECT rp.organization_slug, rp.team_id
    INTO NEW.organization_slug, NEW.team_id
    FROM research_projects rp
   WHERE rp.id = NEW.research_project_id;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS kanban_cards_scope_sync ON kanban_cards;

CREATE TRIGGER kanban_cards_scope_sync
  BEFORE INSERT OR UPDATE OF research_project_id ON kanban_cards
  FOR EACH ROW EXECUTE FUNCTION kanban_cards_fill_scope();

-- Project side: if a project moves to a different organization or team,
-- propagate the new scope to every card under it so the cards move in and out
-- of team / organization board shapes correctly.
CREATE OR REPLACE FUNCTION research_projects_propagate_scope() RETURNS trigger AS $$
BEGIN
  IF NEW.organization_slug IS DISTINCT FROM OLD.organization_slug
     OR NEW.team_id IS DISTINCT FROM OLD.team_id THEN
    UPDATE kanban_cards
       SET organization_slug = NEW.organization_slug,
           team_id           = NEW.team_id
     WHERE research_project_id = NEW.id;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS research_projects_scope_propagate ON research_projects;

CREATE TRIGGER research_projects_scope_propagate
  AFTER UPDATE OF organization_slug, team_id ON research_projects
  FOR EACH ROW EXECUTE FUNCTION research_projects_propagate_scope();

-- Backfill existing cards (idempotent; a no-op on a freshly created database).
UPDATE kanban_cards c
   SET organization_slug = rp.organization_slug,
       team_id           = rp.team_id
  FROM research_projects rp
 WHERE rp.id = c.research_project_id
   AND (c.organization_slug IS DISTINCT FROM rp.organization_slug
        OR c.team_id IS DISTINCT FROM rp.team_id);
