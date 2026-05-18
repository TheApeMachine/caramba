import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import { createCollection } from "@tanstack/react-db";
import { z } from "zod";
import {
	createResearchPaper,
	updateResearchPaper,
} from "#/server/research-papers";

/*
ResearchPaperRow is synced through Electric for the active Clerk organization.
document holds editor state: { metadata, blocks } plus optional extension keys.
revision increments on each server-accepted save; paper_revision_events stores
history (snapshots and optional RFC 6902 patch) for collaborative audit trails.
*/
export const ResearchPaperDocument = z.record(z.string(), z.unknown());

export const ResearchPaperRow = z.object({
	id: z.uuid(),
	research_project_id: z.uuid(),
	organization_slug: z.string(),
	title: z.string(),
	document: z.preprocess((value) => {
		if (typeof value === "string") {
			try {
				return JSON.parse(value) as Record<string, unknown>;
			} catch {
				return {};
			}
		}

		return value;
	}, ResearchPaperDocument),
	revision: z.coerce.number().int().nonnegative(),
	created_at: z.coerce.date(),
	updated_at: z.coerce.date(),
});

export type ResearchPaperRowType = z.infer<typeof ResearchPaperRow>;

const shapeUrl =
	typeof window !== "undefined"
		? `${window.location.origin}/api/shape/research-papers`
		: "/api/shape/research-papers";

export const researchPaperCollection = createCollection(
	electricCollectionOptions({
		id: "research_papers",
		schema: ResearchPaperRow,
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

			const row = transaction.mutations[0].modified as ResearchPaperRowType;

			try {
				const result = await createResearchPaper({
					data: {
						id: row.id,
						research_project_id: row.research_project_id,
						title: row.title,
						document: row.document,
					},
				});

				if (import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true") {
					return;
				}

				if (typeof result?.txid !== "number") {
					console.error(
						"createResearchPaper returned no txid",
						String(import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT),
					);
					return;
				}

				return { timeout: 60_000, txid: result.txid };
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				console.error(`createResearchPaper failed: ${message}`);
				throw err instanceof Error
					? new Error(`createResearchPaper: ${message}`, { cause: err })
					: err;
			}
		},
		onUpdate: async ({ transaction }) => {
			if (!transaction.mutations.length) {
				throw new Error("onUpdate called with no mutations");
			}

			const mutation = transaction.mutations[0];
			const row = mutation.modified as ResearchPaperRowType;
			const original = mutation.original as ResearchPaperRowType | undefined;

			if (original === undefined) {
				throw new Error("research paper update missing prior row");
			}

			const meta = (transaction.metadata ?? {}) as { summary?: string };

			try {
				const result = await updateResearchPaper({
					data: {
						id: row.id,
						expected_revision: original.revision,
						title: row.title,
						document: row.document,
						summary: meta.summary ?? "update",
					},
				});

				if (import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true") {
					return;
				}

				if (typeof result?.txid !== "number") {
					return;
				}

				return { timeout: 60_000, txid: result.txid };
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				console.error(`updateResearchPaper failed: ${message}`);
				throw err instanceof Error
					? new Error(`updateResearchPaper: ${message}`, { cause: err })
					: err;
			}
		},
	}),
);
