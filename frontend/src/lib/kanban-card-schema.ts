import { z } from "zod";

export const kanbanColumnKeySchema = z.enum([
	"backlog",
	"todo",
	"in-progress",
	"review",
	"done",
]);

export type KanbanColumnKey = z.infer<typeof kanbanColumnKeySchema>;

export const kanbanPrioritySchema = z.enum([
	"low",
	"medium",
	"high",
	"critical",
]);

export const kanbanCardLabelSchema = z.object({
	id: z.string(),
	text: z.string(),
	color: z.string(),
});

/** Row shape for the `kanban_cards` table synced via Electric. */
export const kanbanCardRowSchema = z.object({
	id: z.uuid(),
	research_project_id: z.uuid(),
	column_key: kanbanColumnKeySchema,
	sort_order: z.number().int(),
	title: z.string(),
	description: z.string(),
	priority: kanbanPrioritySchema,
	labels_json: z.string(),
	assignees_json: z.string(),
	due_date: z.coerce.date().nullable(),
	requested_by: z.string().nullable(),
	created_at: z.coerce.date(),
	updated_at: z.coerce.date(),
});

export type KanbanCardRow = z.infer<typeof kanbanCardRowSchema>;
