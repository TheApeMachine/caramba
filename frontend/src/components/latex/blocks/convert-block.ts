import type { ResearchPaperBlockRowType } from "#/collections/research_paper_blocks";
import type {
	HeadingLevel,
	PaperBlockKind,
	PaperHeadingPresentation,
} from "#/components/latex/model/types";

export type SetBlockKindOptions = {
	level?: HeadingLevel;
	ordered?: boolean;
	presentation?: PaperHeadingPresentation | null;
};

const readText = (row: ResearchPaperBlockRowType): string => {
	if (row.kind === "equation") {
		return row.latex;
	}

	return row.text;
};

/*
applyBlockKindConversion mutates an Immer draft in place to reflect a
kind change. Used by researchPaperBlockCollection.update(...) handlers
to switch a paragraph to a heading, etc., while preserving carried text
and clearing fields that no longer apply.
*/
export const applyBlockKindConversion = (
	draft: ResearchPaperBlockRowType,
	nextKind: PaperBlockKind,
	options: SetBlockKindOptions = {},
): void => {
	if (draft.kind === nextKind) {
		return;
	}

	const carriedText = readText(draft);

	draft.kind = nextKind;
	draft.text = "";
	draft.latex = "";
	draft.heading_level = null;
	draft.heading_presentation = null;
	draft.list_ordered = false;
	draft.equation_display = true;
	draft.equation_label = "";

	if (nextKind === "heading") {
		const presentation = resolveHeadingPresentation(options.presentation);

		draft.heading_level = options.level ?? 2;
		draft.heading_presentation = presentation;
		draft.text = carriedText;
		return;
	}

	if (nextKind === "equation") {
		draft.latex = carriedText;
		draft.equation_display = true;
		return;
	}

	if (nextKind === "list") {
		draft.list_ordered = options.ordered ?? false;
		draft.text = carriedText;
		return;
	}

	draft.text = carriedText;
};

const resolveHeadingPresentation = (
	presentation: PaperHeadingPresentation | null | undefined,
): string | null => {
	if (presentation === null) {
		return null;
	}

	if (presentation === undefined) {
		return null;
	}

	return presentation;
};
