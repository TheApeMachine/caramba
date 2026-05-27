import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import { createCollection } from "@tanstack/react-db";
import { shapeUrl } from "#/lib/electric-shape";
import { kanbanCardRowSchema } from "#/lib/kanban-card-schema";
import { insertKanbanCard } from "#/server/kanban-cards";

export const kanbanCardsCollection = createCollection(
	electricCollectionOptions({
		id: "kanban_cards",
		schema: kanbanCardRowSchema,
		getKey: (item) => item.id,
		shapeOptions: {
			// Scoped to the caller's organization by the proxy (auth + where clause),
			// like every other collection. Previously this pointed straight at
			// Electric with no where clause and synced the entire table to every
			// client, which is what was dragging out page load.
			url: shapeUrl("kanban-cards"),
			parser: {
				timestamptz: (value: string) => new Date(value),
			},
		},
		onInsert: async ({ transaction }) => {
			const row = transaction.mutations[0].modified;
			const result = await insertKanbanCard({ data: row });

			if (import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true") {
				return;
			}

			return { timeout: 60_000, txid: result.txid };
		},
	}),
);
