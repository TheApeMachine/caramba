import {
	createCollection,
	localStorageCollectionOptions,
} from "@tanstack/react-db";
import { z } from "zod";

/*
ResearchGraphRow is the only persisted shape for the Flume editor.
Everything that needs to survive a reload — nodes, edges (encoded
inside node.connections), comments, viewport — lives here. Ephemeral
view state (drag, hover, selection, routing mode) lives in
flumeEditorStore. No other state mechanism is sanctioned for graph
data: no reducers, no localStorage shadows, no React mirrors.

schema_version bumps on breaking shape changes so older rows can be
migrated or refused at read time.
*/
export const ResearchGraphRow = z.object({
	id: z.string().min(1),
	project_id: z.string().nullable(),
	schema_version: z.number().int().default(1),
	nodes: z.record(z.string(), z.unknown()),
	comments: z.record(z.string(), z.unknown()).default({}),
	viewport: z
		.object({
			scale: z.number().default(1),
			translate: z
				.object({
					x: z.number().default(0),
					y: z.number().default(0),
				})
				.default({ x: 0, y: 0 }),
		})
		.default({ scale: 1, translate: { x: 0, y: 0 } }),
	updated_at: z.coerce.date(),
});

export type ResearchGraphRowType = z.infer<typeof ResearchGraphRow>;

export const researchGraphCollection = createCollection(
	localStorageCollectionOptions({
		id: "research_graphs_local",
		storageKey: "caramba:research_graphs",
		schema: ResearchGraphRow,
		getKey: (item) => item.id,
	}),
);
