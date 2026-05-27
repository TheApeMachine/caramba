import { researchPaperCollection } from "#/collections/research_paper";
import { researchPaperBlockCollection } from "#/collections/research_paper_blocks";
import { serializePaperDocument } from "#/components/latex/model/paper-document";
import type { PaperMetadata } from "#/components/latex/model/types";

const SORT_GAP = 1024;

const buildInitialMetadata = (): PaperMetadata => ({
	title: "",
	authors: "",
	keywords: "",
	abstract: "",
});

const insertInitialBlocks = (paperId: string): void => {
	const now = new Date();
	const headingId = crypto.randomUUID();
	const paragraphId = crypto.randomUUID();

	researchPaperBlockCollection.insert({
		id: headingId,
		paper_id: paperId,
		organization_slug: "",
		sort_order: 0,
		kind: "heading",
		text: "Untitled paper",
		latex: "",
		heading_level: 1,
		heading_presentation: null,
		list_ordered: false,
		equation_display: true,
		equation_label: "",
		created_at: now,
		updated_at: now,
	});

	researchPaperBlockCollection.insert({
		id: paragraphId,
		paper_id: paperId,
		organization_slug: "",
		sort_order: SORT_GAP,
		kind: "paragraph",
		text: "",
		latex: "",
		heading_level: null,
		heading_presentation: null,
		list_ordered: false,
		equation_display: true,
		equation_label: "",
		created_at: now,
		updated_at: now,
	});
};

/*
insertResearchPaperForProject creates a blank paper row plus its initial
heading + paragraph blocks. The paper row carries only metadata; blocks
live in researchPaperBlockCollection so every edit is its own collection
mutation.
*/
export const insertResearchPaperForProject = async (
	researchProjectId: string,
	title: string,
): Promise<string> => {
	const paperId = crypto.randomUUID();
	const now = new Date();
	const trimmedTitle = title.trim();
	const document = serializePaperDocument(buildInitialMetadata());

	const transaction = researchPaperCollection.insert({
		id: paperId,
		research_project_id: researchProjectId,
		organization_slug: "",
		title: trimmedTitle || "Untitled paper",
		document,
		revision: 1,
		created_at: now,
		updated_at: now,
	});

	await transaction.isPersisted.promise;
	insertInitialBlocks(paperId);

	return paperId;
};
