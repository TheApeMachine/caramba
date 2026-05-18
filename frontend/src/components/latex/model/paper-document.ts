import type { PaperBlock, PaperMetadata } from "#/components/latex/model/types";

export type PaperDocumentV1 = {
	metadata: PaperMetadata;
	blocks: PaperBlock[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPaperBlock(value: unknown): value is PaperBlock {
	if (!isRecord(value)) {
		return false;
	}

	const type = value.type;
	if (
		type !== "heading" &&
		type !== "paragraph" &&
		type !== "equation" &&
		type !== "list"
	) {
		return false;
	}

	if (typeof value.id !== "string") {
		return false;
	}

	if (type === "heading") {
		const level = value.level;

		if (level !== 1 && level !== 2 && level !== 3) {
			return false;
		}

		if (typeof value.text !== "string") {
			return false;
		}

		if (value.presentation === undefined) {
			return true;
		}

		const pres = value.presentation;
		return (
			pres === "abstract" || pres === "references" || pres === "acknowledgments"
		);
	}

	if (type === "paragraph") {
		return typeof value.text === "string";
	}

	if (type === "equation") {
		return (
			typeof value.latex === "string" && typeof value.display === "boolean"
		);
	}

	if (type === "list") {
		return typeof value.ordered === "boolean" && typeof value.text === "string";
	}

	return false;
}

/*
parsePaperDocument normalizes persisted JSON into editor state.
Returns null if the payload is unusable.
*/
export function parsePaperDocument(raw: unknown): PaperDocumentV1 | null {
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

	const blocksRaw = raw.blocks;
	if (!Array.isArray(blocksRaw)) {
		return null;
	}

	const blocks: PaperBlock[] = [];

	for (const entry of blocksRaw) {
		if (!isPaperBlock(entry)) {
			return null;
		}

		blocks.push(entry);
	}

	if (blocks.length === 0) {
		return null;
	}

	return { metadata, blocks };
}

/*
serializePaperDocument builds the JSON object stored in research_papers.document.
*/
export function serializePaperDocument(
	metadata: PaperMetadata,
	blocks: PaperBlock[],
): Record<string, unknown> {
	return {
		metadata: {
			title: metadata.title,
			authors: metadata.authors,
			keywords: metadata.keywords,
			abstract: metadata.abstract,
		},
		blocks: blocks.map((block) => ({ ...block })),
	};
}
