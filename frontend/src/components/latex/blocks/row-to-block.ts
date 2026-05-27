import type { ResearchPaperBlockRowType } from "#/collections/research_paper_blocks";
import type {
	HeadingLevel,
	PaperBlock,
	PaperHeadingPresentation,
} from "#/components/latex/model/types";

const ALLOWED_HEADING_LEVELS = new Set<HeadingLevel>([1, 2, 3]);

const ALLOWED_HEADING_PRESENTATIONS = new Set<PaperHeadingPresentation>([
	"abstract",
	"references",
	"acknowledgments",
]);

const headingLevelFromRow = (
	value: number | null | undefined,
): HeadingLevel => {
	if (value === null || value === undefined) {
		return 2;
	}

	const candidate = value as HeadingLevel;

	if (ALLOWED_HEADING_LEVELS.has(candidate)) {
		return candidate;
	}

	return 2;
};

const headingPresentationFromRow = (
	value: string | null | undefined,
): PaperHeadingPresentation | undefined => {
	if (!value) {
		return undefined;
	}

	const candidate = value as PaperHeadingPresentation;

	if (ALLOWED_HEADING_PRESENTATIONS.has(candidate)) {
		return candidate;
	}

	return undefined;
};

/*
researchPaperBlockRowToBlock projects a database row into the in-app
PaperBlock discriminated union the editor and tooling consume. The row
schema carries every field that might apply to any kind; this collapses
them down to the kind-specific shape so the editor's existing renderers
keep working without per-kind branching at every consumer.
*/
export const researchPaperBlockRowToBlock = (
	row: ResearchPaperBlockRowType,
): PaperBlock => {
	if (row.kind === "heading") {
		const presentation = headingPresentationFromRow(row.heading_presentation);

		return {
			id: row.id,
			type: "heading",
			level: headingLevelFromRow(row.heading_level),
			text: row.text,
			...(presentation ? { presentation } : {}),
		};
	}

	if (row.kind === "equation") {
		return {
			id: row.id,
			type: "equation",
			latex: row.latex,
			display: row.equation_display,
			...(row.equation_label ? { label: row.equation_label } : {}),
		};
	}

	if (row.kind === "list") {
		return {
			id: row.id,
			type: "list",
			ordered: row.list_ordered,
			text: row.text,
		};
	}

	return { id: row.id, type: "paragraph", text: row.text };
};
