import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import { createCollection } from "@tanstack/react-db";
import { z } from "zod";
import { createResearchProject } from "#/server/create-research-project";

export const ResearchProject = z.object({
	id: z.uuid(),
	name: z.string().min(1),
	description: z.string(),
	organization_slug: z.string().default(""),
	project_slug: z.preprocess(
		(v) => (v === "" ? null : v),
		z.string().min(1).nullable().optional(),
	),
	created_at: z.coerce.date(),
	updated_at: z.coerce.date(),
});

const shapeUrl =
	typeof window !== "undefined"
		? `${window.location.origin}/api/shape/research-projects`
		: "/api/shape/research-projects";

export const researchProjectCollection = createCollection(
	electricCollectionOptions({
		id: "research_projects",
		schema: ResearchProject,
		getKey: (item) => item.id,
		shapeOptions: {
			url: shapeUrl,
			parser: {
				timestamptz: (value: string) => new Date(value),
			},
		},
		onInsert: async ({ transaction }) => {
			if (!transaction.mutations.length) {
				throw new Error("onInsert called with no mutations");
			}

			const row = transaction.mutations[0].modified;

			try {
				const result = await createResearchProject({ data: row });

				if (import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true") {
					return;
				}

				if (!result?.txid) {
					console.error("createResearchProject returned no txid", result);
					return;
				}

				return { timeout: 60_000, txid: result.txid };
			} catch (err) {
				console.error("createResearchProject failed", err);
				throw err;
			}
		},
	}),
);
