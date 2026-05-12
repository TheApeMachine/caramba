import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import { createCollection } from "@tanstack/react-db";
import { researchProjectRowSchema } from "#/lib/research-project-schema";
import { createResearchProject } from "#/server/create-research-project";

const electricShapeUrl =
	import.meta.env.VITE_ELECTRIC_SHAPE_URL ?? "http://127.0.0.1:3010/v1/shape";

const organizationSlug = import.meta.env.VITE_RESEARCH_PROJECT_ORGANIZATION_SLUG;

if (!organizationSlug) {
	throw new Error("VITE_RESEARCH_PROJECT_ORGANIZATION_SLUG is required");
}

export const researchProjectsCollection = createCollection(
	electricCollectionOptions({
		id: "research_projects",
		schema: researchProjectRowSchema,
		getKey: (item) => item.id,
		shapeOptions: {
			url: electricShapeUrl,
			params: {
				table: "research_projects",
				where: "organization_slug = $1",
				params: JSON.stringify([organizationSlug]),
			},
			parser: {
				timestamptz: (value: string) => new Date(value),
			},
		},
		onInsert: async ({ transaction }) => {
			const row = transaction.mutations[0].modified;
			const result = await createResearchProject({ data: row });
			// Skip waiting for this txid on the Electric stream (dev fallback if replication lags).
			if (import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true") {
				return;
			}
			return { timeout: 60_000, txid: result.txid };
		},
	}),
);
