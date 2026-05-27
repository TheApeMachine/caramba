import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import { createCollection } from "@tanstack/react-db";
import { z } from "zod";
import {
	createResearchPaperBlock,
	deleteResearchPaperBlock,
	reorderResearchPaperBlocks,
	updateResearchPaperBlock,
} from "#/server/research-paper-blocks";

/*
ResearchPaperBlockRow is one block (paragraph / heading / equation / list)
inside a research paper. The Electric shape proxy filters by organization
slug so every signed-in user sees only their org's rows. Editing happens
through TanStack DB collection.insert / update / delete which fire the
mutation handlers below; the paper editor never reaches for a reducer.
*/
export const ResearchPaperBlockRow = z.object({
	id: z.uuid(),
	paper_id: z.uuid(),
	organization_slug: z.string(),
	sort_order: z.preprocess(
		(value) => (typeof value === "bigint" ? Number(value) : value),
		z.number().int().nonnegative(),
	),
	kind: z.enum(["paragraph", "heading", "equation", "list"]),
	text: z.string(),
	latex: z.string(),
	heading_level: z.number().int().min(1).max(3).nullable(),
	heading_presentation: z.string().nullable(),
	list_ordered: z.boolean(),
	equation_display: z.boolean(),
	equation_label: z.string(),
	created_at: z.coerce.date(),
	updated_at: z.coerce.date(),
});

export type ResearchPaperBlockRowType = z.infer<typeof ResearchPaperBlockRow>;

const awaitElectricTxid = (
	result: { txid?: number } | undefined,
): { timeout: number; txid: number } | undefined => {
	if (import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true") {
		return undefined;
	}

	if (typeof result?.txid !== "number") {
		return undefined;
	}

	return { timeout: 60_000, txid: result.txid };
};

const shapeUrl =
	typeof window !== "undefined"
		? `${window.location.origin}/api/shape/research-paper-blocks`
		: "/api/shape/research-paper-blocks";

const upsertPayloadFromRow = (row: ResearchPaperBlockRowType) => ({
	id: row.id,
	paper_id: row.paper_id,
	sort_order: row.sort_order,
	kind: row.kind,
	text: row.text,
	latex: row.latex,
	heading_level: row.kind === "heading" ? row.heading_level : null,
	heading_presentation:
		row.heading_presentation === null
			? undefined
			: (row.heading_presentation as "abstract" | "references" | "acknowledgments"),
	list_ordered: row.list_ordered,
	equation_display: row.equation_display,
	equation_label: row.equation_label,
});

export const researchPaperBlockCollection = createCollection(
	electricCollectionOptions({
		id: "research_paper_blocks",
		schema: ResearchPaperBlockRow,
		getKey: (row) => row.id,
		shapeOptions: {
			url: shapeUrl,
			parser: {
				timestamptz: (value: string) => new Date(value),
			},
		},
		onInsert: async ({ transaction }) => {
			const results = await Promise.all(
				transaction.mutations.map((mutation) =>
					createResearchPaperBlock({
						data: upsertPayloadFromRow(
							mutation.modified as ResearchPaperBlockRowType,
						),
					}),
				),
			);

			return awaitElectricTxid(results[results.length - 1]);
		},
		onUpdate: async ({ transaction }) => {
			const results = await Promise.all(
				transaction.mutations.map((mutation) =>
					updateResearchPaperBlock({
						data: upsertPayloadFromRow(
							mutation.modified as ResearchPaperBlockRowType,
						),
					}),
				),
			);

			return awaitElectricTxid(results[results.length - 1]);
		},
		onDelete: async ({ transaction }) => {
			const results = await Promise.all(
				transaction.mutations.map((mutation) => {
					const original = mutation.original as ResearchPaperBlockRowType;
					return deleteResearchPaperBlock({ data: { id: original.id } });
				}),
			);

			return awaitElectricTxid(results[results.length - 1]);
		},
	}),
);

/*
reorderPaperBlocks is the batch-update path used when the user drags one
block above or below another. The single backend call updates every
affected sort_order in one transaction and returns a single txid the
collection can await.
*/
export const reorderPaperBlocks = async (
	paperId: string,
	entries: ReadonlyArray<{ id: string; sort_order: number }>,
): Promise<void> => {
	if (entries.length === 0) {
		return;
	}

	await reorderResearchPaperBlocks({
		data: {
			paper_id: paperId,
			entries: entries.map((entry) => ({
				id: entry.id,
				sort_order: entry.sort_order,
			})),
		},
	});
};
