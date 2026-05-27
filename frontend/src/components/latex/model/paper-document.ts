import type { PaperMetadata } from "#/components/latex/model/types";

/*
PaperDocumentV2 carries only the paper's metadata. Blocks moved out of
this JSON blob into research_paper_blocks rows in v2 — every block is
its own Tanstack DB collection entry now, so the editor never round-trips
the whole document on a single edit.
*/
export type PaperDocumentV2 = {
	metadata: PaperMetadata;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
	typeof value === "object" && value !== null && !Array.isArray(value);

/*
parsePaperDocument extracts the metadata block from the JSON stored in
research_papers.document. Falls back to an empty metadata object on any
shape mismatch so a malformed document never crashes the editor.
*/
export const parsePaperDocument = (raw: unknown): PaperDocumentV2 | null => {
	if (!isRecord(raw)) {
		return null;
	}

	const metaRaw = raw.metadata;

	if (!isRecord(metaRaw)) {
		return null;
	}

	const metadata: PaperMetadata = {
		title: typeof metaRaw.title === "string" ? metaRaw.title : "",
		authors: typeof metaRaw.authors === "string" ? metaRaw.authors : "",
		keywords: typeof metaRaw.keywords === "string" ? metaRaw.keywords : "",
		abstract: typeof metaRaw.abstract === "string" ? metaRaw.abstract : "",
	};

	return { metadata };
};

/*
serializePaperDocument builds the JSON object stored in research_papers.document.
Only metadata lives in this blob; blocks are persisted independently.
*/
export const serializePaperDocument = (
	metadata: PaperMetadata,
): Record<string, unknown> => ({
	metadata: {
		title: metadata.title,
		authors: metadata.authors,
		keywords: metadata.keywords,
		abstract: metadata.abstract,
	},
});
