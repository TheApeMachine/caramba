/*
Jazz 2.0 schema — the synced data model (source of truth in the CRDT-primary
design). The Jazz dev plugin (jazz-tools/dev/vite) watches this file and pushes
the structural schema to the sync server. A server-side bridge worker
(jazz-tools/backend) projects these tables into Postgres so the Go orchestrator
+ pg_notify keep working.

Column DSL verified against jazz-tools@2.0.0-alpha.50 (dist/dsl.d.ts):
  s.string() s.boolean() s.int() s.float() s.timestamp() s.json()
  s.enum(...variants) s.ref("table") s.bytes() s.array(el)  + .optional() .default()

Note: Jazz resolves concurrent same-field writes last-writer-wins. For live
co-typing in the same paragraph, paperBlocks.text/latex should be backed by a
text-CRDT (Yjs/Loro) in Phase 1 — this row carries structure, the text-CRDT
carries characters.
*/

import { schema as s } from "jazz-tools";

export const app = s.defineApp({
	teams: s.table({
		organization_slug: s.string(),
		name: s.string(),
		slug: s.string(),
		description: s.string(),
		privacy_mode: s.enum("shared", "local"),
		color: s.string(),
		emoji: s.string(),
		created_at: s.timestamp(),
		updated_at: s.timestamp(),
	}),

	teamMemberships: s.table({
		team: s.ref("teams"),
		user_id: s.string(),
		role: s.enum("owner", "member"),
		created_at: s.timestamp(),
	}),

	projects: s.table({
		name: s.string(),
		description: s.string(),
		organization_slug: s.string(),
		project_slug: s.string().optional(),
		team: s.ref("teams").optional(),
		created_at: s.timestamp(),
		updated_at: s.timestamp(),
	}),

	projectMembers: s.table({
		project: s.ref("projects"),
		user_id: s.string(),
		role: s.enum("owner", "member"),
		created_at: s.timestamp(),
	}),

	kanbanCards: s.table({
		project: s.ref("projects"),
		organization_slug: s.string(),
		team: s.ref("teams").optional(),
		column_key: s.enum("backlog", "todo", "in-progress", "review", "done"),
		sort_order: s.int(),
		title: s.string(),
		description: s.string(),
		priority: s.enum("low", "medium", "high", "critical"),
		labels: s.json(),
		assignees: s.json(),
		due_date: s.timestamp().optional(),
		requested_by: s.string().optional(),
		created_at: s.timestamp(),
		updated_at: s.timestamp(),
	}),

	kanbanSubtasks: s.table({
		card: s.ref("kanbanCards"),
		sort_order: s.int(),
		title: s.string(),
		description: s.string(),
		status: s.enum("todo", "in-progress", "done", "failed"),
		assigned_agent: s.string().optional(),
		context: s.json(),
		created_at: s.timestamp(),
		updated_at: s.timestamp(),
	}),

	papers: s.table({
		project: s.ref("projects"),
		organization_slug: s.string(),
		title: s.string(),
		document: s.json(),
		revision: s.int(),
		created_at: s.timestamp(),
		updated_at: s.timestamp(),
	}),

	paperBlocks: s.table({
		paper: s.ref("papers"),
		organization_slug: s.string(),
		sort_order: s.int(),
		kind: s.enum("paragraph", "heading", "list", "equation"),
		text: s.string(),
		latex: s.string(),
		heading_level: s.int().optional(),
		heading_presentation: s
			.enum("abstract", "references", "acknowledgments")
			.optional(),
		list_ordered: s.boolean(),
		equation_display: s.boolean(),
		equation_label: s.string(),
		created_at: s.timestamp(),
		updated_at: s.timestamp(),
	}),

	// Phase 4: assistant_personas / assistant_sessions / assistant_session_personas
	// / assistant_messages move here once kanban + projects are proven.
});
