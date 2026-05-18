import { researchPaperCollection } from "#/collections/research_paper";
import { serializePaperDocument } from "#/components/latex/model/paper-document";
import { createInitialPaperBlocks } from "#/components/latex/model/paper-reducer";
import type { PaperMetadata } from "#/components/latex/model/types";

/*
insertResearchPaperForProject creates a blank paper row linked to a research project.
Additional papers can be added any time after provisioning — one project may publish many.
*/
export const insertResearchPaperForProject = async (
	researchProjectId: string,
	title: string,
): Promise<string> => {
	const paperId = crypto.randomUUID();
	const initialBlocks = createInitialPaperBlocks();
	const initialMetadata: PaperMetadata = {
		title: "",
		authors: "",
		keywords: "",
		abstract: "",
	};
	const document = serializePaperDocument(initialMetadata, initialBlocks);
	const now = new Date();
	const trimmedTitle = title.trim();

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

	return paperId;
};
