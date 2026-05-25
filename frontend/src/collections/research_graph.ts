import {
	createCollection,
	localStorageCollectionOptions,
} from "@tanstack/react-db";
import { z } from "zod";

/*
ResearchGraphRow stores the Flume NodeMap as JSON-serializable graph state.
One row per project (id matches project_id when bound to a research project).
*/
export const ResearchGraphRow = z.object({
	id: z.string().min(1),
	project_id: z.string().nullable(),
	nodes: z.record(z.string(), z.unknown()),
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
