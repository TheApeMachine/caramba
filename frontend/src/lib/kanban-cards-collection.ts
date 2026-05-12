import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import { createCollection } from "@tanstack/react-db";
import { kanbanCardRowSchema } from "#/lib/kanban-card-schema";
import { insertKanbanCard } from "#/server/kanban-cards";

const electricShapeUrl =
	import.meta.env.VITE_ELECTRIC_SHAPE_URL ?? "http://127.0.0.1:3010/v1/shape";

export const kanbanCardsCollection = createCollection(
	electricCollectionOptions({
		id: "kanban_cards",
		schema: kanbanCardRowSchema,
		getKey: (item) => item.id,
		shapeOptions: {
			url: electricShapeUrl,
			params: { table: "kanban_cards" },
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
