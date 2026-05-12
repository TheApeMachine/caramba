import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import { createCollection } from "@tanstack/react-db";
import { z } from "zod";
import { createResearchProject } from "#/server/create-research-project";

export const ResearchProject = z.object({
	id: z.uuid(),
	name: z.string().min(1),
	description: z.string(),
	organization_slug: z.string().default(""),
	project_slug: z.string().min(1).nullable().optional(),
	created_at: z.coerce.date(),
	updated_at: z.coerce.date(),
});

export const researchProjectCollection = createCollection(
	electricCollectionOptions({
		id: "research_projects",
		schema: ResearchProject,
		getKey: (item) => item.id,
		shapeOptions: {
			url: `${window.location.origin}/api/shape/research-projects`,
			parser: {
				timestamptz: (value: string) => new Date(value),
			},
		},
		onInsert: async ({ transaction }) => {
			const row = transaction.mutations[0].modified;
			const result = await createResearchProject({ data: row });

			if (import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true") {
				return;
			}

			return { timeout: 60_000, txid: result.txid };
		},
	}),
);
