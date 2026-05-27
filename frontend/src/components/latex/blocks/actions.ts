"use client";

import {
	reorderPaperBlocks,
	type ResearchPaperBlockRowType,
	researchPaperBlockCollection,
} from "#/collections/research_paper_blocks";
import {
	applyBlockKindConversion,
	type SetBlockKindOptions,
} from "#/components/latex/blocks/convert-block";
import type {
	HeadingLevel,
	PaperBlock,
	PaperBlockKind,
} from "#/components/latex/model/types";

const SORT_GAP = 1024;

const nowTimestamp = (): Date => new Date();

const insertOrder = (
	blocks: ReadonlyArray<ResearchPaperBlockRowType>,
	afterId: string,
): number => {
	const sorted = [...blocks].sort((left, right) =>
		left.sort_order === right.sort_order
			? left.created_at.getTime() - right.created_at.getTime()
			: left.sort_order - right.sort_order,
	);

	const index = sorted.findIndex((entry) => entry.id === afterId);

	if (index === -1) {
		const last = sorted[sorted.length - 1];
		return (last ? last.sort_order : 0) + SORT_GAP;
	}

	const current = sorted[index];
	const next = sorted[index + 1];

	if (!next) {
		return current.sort_order + SORT_GAP;
	}

	const midpoint = (current.sort_order + next.sort_order) / 2;

	if (midpoint === current.sort_order || midpoint === next.sort_order) {
		return current.sort_order + SORT_GAP;
	}

	return midpoint;
};

const baseRow = (params: {
	id: string;
	paperId: string;
	organizationSlug: string;
	sortOrder: number;
	kind: PaperBlockKind;
}): ResearchPaperBlockRowType => {
	const now = nowTimestamp();

	return {
		id: params.id,
		paper_id: params.paperId,
		organization_slug: params.organizationSlug,
		sort_order: params.sortOrder,
		kind: params.kind,
		text: "",
		latex: "",
		heading_level: null,
		heading_presentation: null,
		list_ordered: false,
		equation_display: true,
		equation_label: "",
		created_at: now,
		updated_at: now,
	};
};

export type InsertContext = {
	paperId: string;
	organizationSlug: string;
	blocks: ReadonlyArray<ResearchPaperBlockRowType>;
};

export const insertParagraphAfter = (
	context: InsertContext,
	afterId: string,
	text: string,
): string => {
	const id = crypto.randomUUID();
	const row = baseRow({
		id,
		paperId: context.paperId,
		organizationSlug: context.organizationSlug,
		sortOrder: insertOrder(context.blocks, afterId),
		kind: "paragraph",
	});
	row.text = text;
	researchPaperBlockCollection.insert(row);
	return id;
};

export const insertHeadingAfter = (
	context: InsertContext,
	afterId: string,
	level: HeadingLevel,
): string => {
	const id = crypto.randomUUID();
	const row = baseRow({
		id,
		paperId: context.paperId,
		organizationSlug: context.organizationSlug,
		sortOrder: insertOrder(context.blocks, afterId),
		kind: "heading",
	});
	row.heading_level = level;
	researchPaperBlockCollection.insert(row);
	return id;
};

export const insertEquationAfter = (
	context: InsertContext,
	afterId: string,
	latex: string,
): string => {
	const id = crypto.randomUUID();
	const row = baseRow({
		id,
		paperId: context.paperId,
		organizationSlug: context.organizationSlug,
		sortOrder: insertOrder(context.blocks, afterId),
		kind: "equation",
	});
	row.latex = latex;
	researchPaperBlockCollection.insert(row);
	return id;
};

export const insertListAfter = (
	context: InsertContext,
	afterId: string,
	ordered: boolean,
): string => {
	const id = crypto.randomUUID();
	const row = baseRow({
		id,
		paperId: context.paperId,
		organizationSlug: context.organizationSlug,
		sortOrder: insertOrder(context.blocks, afterId),
		kind: "list",
	});
	row.list_ordered = ordered;
	researchPaperBlockCollection.insert(row);
	return id;
};

export const insertBlockAfter = (
	context: InsertContext,
	afterId: string,
	block: PaperBlock,
): string => {
	const sortOrder = insertOrder(context.blocks, afterId);
	const row = baseRow({
		id: block.id,
		paperId: context.paperId,
		organizationSlug: context.organizationSlug,
		sortOrder,
		kind: block.type,
	});

	if (block.type === "heading") {
		row.heading_level = block.level;
		row.text = block.text;
		row.heading_presentation = block.presentation ?? null;
	} else if (block.type === "equation") {
		row.latex = block.latex;
		row.equation_display = block.display;
		row.equation_label = block.label ?? "";
	} else if (block.type === "list") {
		row.list_ordered = block.ordered;
		row.text = block.text;
	} else {
		row.text = block.text;
	}

	researchPaperBlockCollection.insert(row);
	return block.id;
};

export const updateBlockText = (id: string, text: string): void => {
	researchPaperBlockCollection.update(id, (draft) => {
		if (draft.kind === "equation") {
			return;
		}

		draft.text = text;
		draft.updated_at = nowTimestamp();
	});
};

export const updateBlockLatex = (id: string, latex: string): void => {
	researchPaperBlockCollection.update(id, (draft) => {
		if (draft.kind !== "equation") {
			return;
		}

		draft.latex = latex;
		draft.updated_at = nowTimestamp();
	});
};

export const setBlockKind = (
	id: string,
	kind: PaperBlockKind,
	options?: SetBlockKindOptions,
): void => {
	researchPaperBlockCollection.update(id, (draft) => {
		applyBlockKindConversion(
			draft as unknown as ResearchPaperBlockRowType,
			kind,
			options,
		);
		(draft as { updated_at: Date }).updated_at = nowTimestamp();
	});
};

export const insertBlockAtStart = (
	context: InsertContext,
	block: PaperBlock,
): string => {
	const sorted = [...context.blocks].sort((left, right) =>
		left.sort_order === right.sort_order
			? left.created_at.getTime() - right.created_at.getTime()
			: left.sort_order - right.sort_order,
	);
	const first = sorted[0];
	const sortOrder = first ? first.sort_order - SORT_GAP : 0;
	const row = baseRow({
		id: block.id,
		paperId: context.paperId,
		organizationSlug: context.organizationSlug,
		sortOrder,
		kind: block.type,
	});

	if (block.type === "heading") {
		row.heading_level = block.level;
		row.text = block.text;
		row.heading_presentation = block.presentation ?? null;
	} else if (block.type === "equation") {
		row.latex = block.latex;
		row.equation_display = block.display;
		row.equation_label = block.label ?? "";
	} else if (block.type === "list") {
		row.list_ordered = block.ordered;
		row.text = block.text;
	} else {
		row.text = block.text;
	}

	researchPaperBlockCollection.insert(row);
	return block.id;
};

export const removeBlock = (
	context: InsertContext,
	id: string,
): void => {
	if (context.blocks.length <= 1) {
		return;
	}

	researchPaperBlockCollection.delete(id);
};

export const reorderBlock = async (
	context: InsertContext,
	sourceId: string,
	targetId: string,
	position: "above" | "below",
): Promise<void> => {
	if (sourceId === targetId) {
		return;
	}

	const sorted = [...context.blocks].sort((left, right) =>
		left.sort_order === right.sort_order
			? left.created_at.getTime() - right.created_at.getTime()
			: left.sort_order - right.sort_order,
	);

	const sourceIndex = sorted.findIndex((entry) => entry.id === sourceId);
	const targetIndex = sorted.findIndex((entry) => entry.id === targetId);

	if (sourceIndex === -1 || targetIndex === -1) {
		return;
	}

	const next = sorted.slice();
	const [moved] = next.splice(sourceIndex, 1);
	const adjusted =
		sourceIndex < targetIndex ? targetIndex - 1 : targetIndex;
	const insertAt = position === "below" ? adjusted + 1 : adjusted;
	next.splice(insertAt, 0, moved);

	const entries = next.map((entry, index) => ({
		id: entry.id,
		sort_order: index * SORT_GAP,
	}));

	await reorderPaperBlocks(context.paperId, entries);
};
