import { app } from "../../schema";
import type { KanbanBoard, KanbanCard } from "#/components/kanban/model";
import { DEFAULT_BOARD } from "#/components/kanban/model";
import { type KanbanColumnKey, kanbanColumnKeySchema } from "#/lib/kanban-card-schema";

/*
KanbanCardJazzRow is the row shape returned by useAll(app.kanbanCards...),
derived straight from the schema so it can never drift. labels/assignees are
native JSON (s.json() -> JsonValue); due_date/team/requested_by are nullable.
*/
type RowOf<TTable> = TTable extends { readonly _rowType: infer TRow }
	? TRow
	: never;

export type KanbanCardJazzRow = RowOf<typeof app.kanbanCards>;

/*
parseKanbanLabels validates an already-parsed JSON value into label chips. Jazz
hands back the stored array directly, so there is no JSON.parse here.
*/
function parseKanbanLabels(value: unknown): KanbanCard["labels"] {
	if (!Array.isArray(value)) {
		return [];
	}

	const labels: KanbanCard["labels"] = [];

	for (const item of value) {
		if (
			typeof item === "object" &&
			item !== null &&
			"id" in item &&
			"text" in item &&
			"color" in item &&
			typeof (item as { id: unknown }).id === "string"
		) {
			labels.push({
				id: (item as { id: string }).id,
				text: String((item as { text: unknown }).text),
				color: String((item as { color: unknown }).color),
			});
		}
	}

	return labels;
}

function parseKanbanAssignees(value: unknown): KanbanCard["assignees"] {
	if (!Array.isArray(value)) {
		return [];
	}

	return value.filter((entry): entry is string => typeof entry === "string");
}

/*
kanbanBoardFromRows materializes column/card structures expected by the Kanban UI.
*/
export function kanbanBoardFromRows(
	rows: KanbanCardJazzRow[],
	projectsById: Map<string, { name: string }>,
	aggregateMode: boolean,
): KanbanBoard {
	const columns = DEFAULT_BOARD.columns.map((column) => ({
		...column,
		cardIds: [] as string[],
	}));

	const cards: Record<string, KanbanCard> = {};

	for (const columnTemplate of DEFAULT_BOARD.columns) {
		const columnRows = rows
			.filter((row) => row.column_key === columnTemplate.id)
			.sort((left, right) => left.sort_order - right.sort_order);

		const column = columns.find((entry) => entry.id === columnTemplate.id);

		if (!column) {
			continue;
		}

		for (const row of columnRows) {
			const projectRecord = projectsById.get(row.project);

			const sourceProjectName =
				aggregateMode && projectRecord !== undefined
					? projectRecord.name
					: undefined;

			const dueDate =
				row.due_date !== null ? row.due_date.toISOString().slice(0, 10) : null;

			cards[row.id] = {
				id: row.id,
				title: row.title,
				description: row.description,
				priority: row.priority,
				labels: parseKanbanLabels(row.labels),
				assignees: parseKanbanAssignees(row.assignees),
				dueDate,
				columnId: row.column_key,
				order: row.sort_order,
				createdAt: row.created_at.toISOString(),
				researchProjectId: row.project,
				sourceProjectName,
			};
			column.cardIds.push(row.id);
		}
	}

	return { columns, cards };
}

/*
collectOrderingUpdates flattens board column order into persistence rows.
*/
export function collectOrderingUpdates(board: KanbanBoard): Array<{
	id: string;
	column_key: KanbanColumnKey;
	sort_order: number;
}> {
	const updates: Array<{
		id: string;
		column_key: KanbanColumnKey;
		sort_order: number;
	}> = [];

	for (const column of board.columns) {
		column.cardIds.forEach((cardId, index) => {
			updates.push({
				id: cardId,
				column_key: kanbanColumnKeySchema.parse(column.id),
				sort_order: index,
			});
		});
	}

	return updates;
}
