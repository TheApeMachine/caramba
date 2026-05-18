export type HeadingLevel = 1 | 2 | 3;

/*
PaperHeadingPresentation tweaks editor layout and LaTeX export so sections
read like their printed form (abstract block, unnumbered back matter, etc.).
*/
export type PaperHeadingPresentation =
	| "abstract"
	| "references"
	| "acknowledgments";

export type PaperHeadingBlock = {
	id: string;
	type: "heading";
	level: HeadingLevel;
	text: string;
	presentation?: PaperHeadingPresentation;
};

export type PaperParagraphBlock = {
	id: string;
	type: "paragraph";
	text: string;
};

export type PaperEquationBlock = {
	id: string;
	type: "equation";
	latex: string;
	label?: string;
	display: boolean;
};

export type PaperListBlock = {
	id: string;
	type: "list";
	ordered: boolean;
	text: string;
};

export type PaperBlock =
	| PaperHeadingBlock
	| PaperParagraphBlock
	| PaperEquationBlock
	| PaperListBlock;

export type PaperBlockKind = PaperBlock["type"];

export type PaperMetadata = {
	title: string;
	authors: string;
	keywords: string;
	abstract: string;
};
