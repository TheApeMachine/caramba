export type HeadingLevel = 1 | 2 | 3;

export type PaperHeadingBlock = {
	id: string;
	type: "heading";
	level: HeadingLevel;
	text: string;
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

export type PaperBlock = PaperHeadingBlock | PaperParagraphBlock | PaperEquationBlock;

export type PaperMetadata = {
	title: string;
	authors: string;
	keywords: string;
	abstract: string;
};
