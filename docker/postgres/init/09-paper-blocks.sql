-- Research paper blocks: every heading / paragraph / equation / list inside a
-- paper is one row in this table. The paper row's document column now carries
-- only metadata (title, authors, keywords, abstract); the blocks live here so
-- TanStack DB collection mutations (insert / update / delete / reorder) drive
-- the editor directly instead of shipping the whole document blob on every
-- keystroke. organization_slug is denormalized so the Electric shape proxy
-- can filter by org without a join.

CREATE TABLE IF NOT EXISTS research_paper_blocks (
  id UUID PRIMARY KEY,
  paper_id UUID NOT NULL REFERENCES research_papers (id) ON DELETE CASCADE,
  organization_slug TEXT NOT NULL DEFAULT '',
  sort_order INTEGER NOT NULL DEFAULT 0,
  kind TEXT NOT NULL,
  text TEXT NOT NULL DEFAULT '',
  latex TEXT NOT NULL DEFAULT '',
  heading_level SMALLINT,
  heading_presentation TEXT,
  list_ordered BOOLEAN NOT NULL DEFAULT FALSE,
  equation_display BOOLEAN NOT NULL DEFAULT TRUE,
  equation_label TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT research_paper_blocks_kind_chk CHECK (
    kind IN ('paragraph', 'heading', 'list', 'equation')
  ),
  CONSTRAINT research_paper_blocks_heading_level_chk CHECK (
    kind <> 'heading' OR (heading_level BETWEEN 1 AND 3)
  ),
  CONSTRAINT research_paper_blocks_heading_presentation_chk CHECK (
    heading_presentation IS NULL OR heading_presentation IN (
      'abstract', 'references', 'acknowledgments'
    )
  )
);

CREATE INDEX IF NOT EXISTS research_paper_blocks_paper_order_idx
  ON research_paper_blocks (paper_id, sort_order);

CREATE INDEX IF NOT EXISTS research_paper_blocks_organization_idx
  ON research_paper_blocks (organization_slug);

ALTER TABLE research_paper_blocks REPLICA IDENTITY FULL;

-- One-time backfill: for every paper that has blocks embedded in its document
-- JSON and no rows yet in research_paper_blocks, materialize each block as a
-- row and strip the blocks array from the document column. Idempotent — the
-- NOT EXISTS guard skips papers that have already been migrated.
DO $$
DECLARE
  paper_row RECORD;
  block_data JSONB;
  block_index INT;
  block_id UUID;
BEGIN
  FOR paper_row IN
    SELECT id, organization_slug, document
      FROM research_papers
     WHERE jsonb_typeof(document->'blocks') = 'array'
       AND NOT EXISTS (
         SELECT 1 FROM research_paper_blocks WHERE paper_id = research_papers.id
       )
  LOOP
    block_index := 0;

    FOR block_data IN
      SELECT value
        FROM jsonb_array_elements(paper_row.document->'blocks')
    LOOP
      IF block_data->>'id' ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$' THEN
        block_id := (block_data->>'id')::uuid;
      ELSE
        block_id := gen_random_uuid();
      END IF;

      INSERT INTO research_paper_blocks (
        id,
        paper_id,
        organization_slug,
        sort_order,
        kind,
        text,
        latex,
        heading_level,
        heading_presentation,
        list_ordered,
        equation_display,
        equation_label
      )
      VALUES (
        block_id,
        paper_row.id,
        paper_row.organization_slug,
        block_index,
        block_data->>'type',
        COALESCE(block_data->>'text', ''),
        COALESCE(block_data->>'latex', ''),
        NULLIF(block_data->>'level', '')::smallint,
        NULLIF(block_data->>'presentation', ''),
        COALESCE((block_data->>'ordered')::boolean, FALSE),
        COALESCE((block_data->>'display')::boolean, TRUE),
        COALESCE(block_data->>'label', '')
      );

      block_index := block_index + 1;
    END LOOP;

    UPDATE research_papers
       SET document = jsonb_build_object(
             'metadata',
             COALESCE(paper_row.document->'metadata', '{}'::jsonb)
           )
     WHERE id = paper_row.id;
  END LOOP;
END
$$;
