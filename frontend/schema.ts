/*
Jazz 2.0 schema — the source of truth for the synced data model.

This maps the existing Postgres entities onto Jazz tables. The Jazz dev plugin
(jazz-tools/dev/vite) watches this file and pushes the structural schema to the
sync server on change. The Postgres init scripts remain the projection target:
a server-side bridge worker subscribes to these tables and writes them back into
Postgres so the Go orchestrator + pg_notify keep working unchanged.

DSL confirmed against the Jazz 2.0 docs:
  s.table({ ... }), s.string(), s.boolean(), s.number(),
  s.ref("table"), .optional(), s.defineApp({ ... })

Conservative choices to verify against the installed jazz-tools@alpha:
  - timestamps are modeled as ISO strings (s.string()) until s.date/s.timestamp
    is confirmed in the installed build.
  - JSON-ish columns (labels, assignees, context, paper document metadata) keep
    the existing "stringified JSON in a TEXT column" representation as s.string().
  - owner_id / user_id / organization_slug stay as plain string columns because
    the permissions layer compares them against the session (see permissions.ts).
*/

import { schema as s } from "jazz-tools";

export const app = s.defineApp({
	teams: s.table({
		organization_slug: s.string(),
		name: s.string(),
		slug: s.string(),
		description: s.string(),
		privacy_mode: s.string(), // 'shared' | 'local'
		color: s.string(),
		emoji: s.string(),
		created_at: s.string(),
		updated_at: s.string(),
	}),

	teamMemberships: s.table({
		team: s.ref("teams"),
		user_id: s.string(),
		role: s.string(), // 'owner' | 'member'
		created_at: s.string(),
	}),

	projects: s.table({
		name: s.string(),
		description: s.string(),
		organization_slug: s.string(),
		project_slug: s.string().optional(),
		team: s.ref("teams").optional(),
		created_at: s.string(),
		updated_at: s.string(),
	}),

	projectMembers: s.table({
		project: s.ref("projects"),
		user_id: s.string(),
		role: s.string(), // 'owner' | 'member'
		created_at: s.string(),
	}),

	kanbanCards: s.table({
		project: s.ref("projects"),
		organization_slug: s.string(),
		team: s.ref("teams").optional(),
		column_key: s.string(), // backlog|todo|in-progress|review|done
		sort_order: s.number(),
		title: s.string(),
		description: s.string(),
		priority: s.string(), // low|medium|high|critical
		labels_json: s.string(),
		assignees_json: s.string(),
		due_date: s.string().optional(),
		requested_by: s.string().optional(),
		created_at: s.string(),
		updated_at: s.string(),
	}),

	kanbanSubtasks: s.table({
		card: s.ref("kanbanCards"),
		sort_order: s.number(),
		title: s.string(),
		description: s.string(),
		status: s.string(), // todo|in-progress|done|failed
		assigned_agent: s.string().optional(),
		context_snapshot: s.string(), // stringified JSON
		created_at: s.string(),
		updated_at: s.string(),
	}),

	papers: s.table({
		project: s.ref("projects"),
		organization_slug: s.string(),
		title: s.string(),
		document: s.string(), // stringified JSON: { metadata }
		revision: s.number(),
		created_at: s.string(),
		updated_at: s.string(),
	}),

	/*
	Paper blocks. Note Jazz 2.0 resolves concurrent writes to the same field with
	last-writer-wins, NOT character-level text merge. For true simultaneous prose
	editing, `text`/`latex` should be backed by a text-CRDT (Yjs/Loro) layer; this
	row carries the structural block, the text-CRDT carries the live characters.
	Tracked as a follow-up — see the migration notes.
	*/
	paperBlocks: s.table({
		paper: s.ref("papers"),
		organization_slug: s.string(),
		sort_order: s.number(),
		kind: s.string(), // paragraph|heading|list|equation
		text: s.string(),
		latex: s.string(),
		heading_level: s.number().optional(),
		heading_presentation: s.string().optional(),
		list_ordered: s.boolean(),
		equation_display: s.boolean(),
		equation_label: s.string(),
		created_at: s.string(),
		updated_at: s.string(),
	}),

	// Phase 4: assistant_personas / assistant_sessions / assistant_session_personas
	// / assistant_messages move here once the kanban + projects surfaces are proven.
});
