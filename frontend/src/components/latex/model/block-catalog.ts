import type {
	HeadingLevel,
	PaperBlock,
	PaperHeadingPresentation,
} from "#/components/latex/model/types";

export type BlockKindDescriptor = {
	id: string;
	label: string;
	category: string;
	hint: string;
	keywords: string[];
	shortcut?: string;
	build: () => PaperBlock;
};

function makeId(): string {
	return crypto.randomUUID();
}

function paperSectionHeading(
	id: string,
	label: string,
	hint: string,
	keywords: string[],
	config?: {
		level?: HeadingLevel;
		presentation?: PaperHeadingPresentation;
	},
): BlockKindDescriptor {
	const level = config?.level ?? 2;
	const presentation = config?.presentation;

	return {
		id,
		label,
		category: "Paper sections",
		hint,
		keywords: [label.toLowerCase(), ...keywords],
		build: () => ({
			id: makeId(),
			type: "heading",
			level,
			text: label,
			...(presentation ? { presentation } : {}),
		}),
	};
}

const paperSectionDescriptors: BlockKindDescriptor[] = [
	paperSectionHeading(
		"sec-abstract",
		"Abstract",
		"Standalone summary",
		["abstract", "summary"],
		{ presentation: "abstract" },
	),
	paperSectionHeading("sec-intro", "Introduction", "Problem and motivation", [
		"introduction",
		"intro",
		"motivation",
	]),
	paperSectionHeading(
		"sec-related",
		"Related work",
		"Positioning against prior art",
		["related", "literature", "review", "prior", "art", "background"],
	),
	paperSectionHeading(
		"sec-background",
		"Background",
		"Preliminaries and setup",
		["background", "preliminaries"],
	),
	paperSectionHeading("sec-methods", "Methods", "Approach and setup", [
		"methods",
		"methodology",
		"approach",
	]),
	paperSectionHeading(
		"sec-experiments",
		"Experiments",
		"Protocol and baselines",
		["experiments", "evaluation", "setup"],
	),
	paperSectionHeading("sec-results", "Results", "Empirical findings", [
		"results",
		"findings",
	]),
	paperSectionHeading(
		"sec-analysis",
		"Analysis",
		"Interpretation and ablations",
		["analysis", "ablation"],
	),
	paperSectionHeading("sec-discussion", "Discussion", "Interpretation", [
		"discussion",
	]),
	paperSectionHeading("sec-limitations", "Limitations", "Scope and caveats", [
		"limitations",
		"caveats",
	]),
	paperSectionHeading("sec-conclusion", "Conclusion", "Wrap-up", [
		"conclusion",
	]),
	paperSectionHeading("sec-future", "Future work", "Open problems", [
		"future",
		"outlook",
		"next",
	]),
	paperSectionHeading(
		"sec-ack",
		"Acknowledgments",
		"Contributors and support",
		["acknowledgments", "thanks", "funding"],
		{ presentation: "acknowledgments" },
	),
	paperSectionHeading(
		"sec-refs",
		"References",
		"Heading before bibliography",
		["references", "bibliography", "citations"],
		{ presentation: "references" },
	),
	paperSectionHeading("sec-appendix", "Appendix", "Deferred material", [
		"appendix",
		"supplementary",
	]),
	paperSectionHeading(
		"sec-contributions",
		"Contributions",
		"Summary of novelty (often early)",
		["contributions", "novelty", "claims"],
	),
];

export const blockCatalog: BlockKindDescriptor[] = [
	...paperSectionDescriptors,
	{
		id: "h1",
		label: "Heading 1",
		category: "Headings",
		hint: "Section",
		keywords: ["heading", "h1", "section", "title"],
		shortcut: "# ",
		build: () => ({ id: makeId(), type: "heading", level: 1, text: "" }),
	},
	{
		id: "h2",
		label: "Heading 2",
		category: "Headings",
		hint: "Subsection",
		keywords: ["heading", "h2", "subsection"],
		shortcut: "## ",
		build: () => ({ id: makeId(), type: "heading", level: 2, text: "" }),
	},
	{
		id: "h3",
		label: "Heading 3",
		category: "Headings",
		hint: "Subsubsection",
		keywords: ["heading", "h3", "subsubsection"],
		shortcut: "### ",
		build: () => ({ id: makeId(), type: "heading", level: 3, text: "" }),
	},
	{
		id: "p",
		label: "Paragraph",
		category: "Text",
		hint: "Body text",
		keywords: ["paragraph", "text", "body"],
		build: () => ({ id: makeId(), type: "paragraph", text: "" }),
	},
	{
		id: "eq",
		label: "Equation",
		category: "Math",
		hint: "Display math",
		keywords: ["equation", "math", "latex", "formula"],
		shortcut: "$$",
		build: () => ({ id: makeId(), type: "equation", latex: "", display: true }),
	},
	{
		id: "ul",
		label: "Bullet list",
		category: "Lists",
		hint: "Unordered",
		keywords: ["list", "bullet", "unordered", "itemize"],
		shortcut: "- ",
		build: () => ({ id: makeId(), type: "list", ordered: false, text: "" }),
	},
	{
		id: "ol",
		label: "Numbered list",
		category: "Lists",
		hint: "Ordered",
		keywords: ["list", "numbered", "ordered", "enumerate"],
		shortcut: "1. ",
		build: () => ({ id: makeId(), type: "list", ordered: true, text: "" }),
	},
];

export type BlockCatalogGroup = {
	label: string;
	items: readonly BlockKindDescriptor[];
};

const categoryOrder = ["Paper sections", "Headings", "Text", "Lists", "Math"];

export const blockCatalogGroups: BlockCatalogGroup[] = categoryOrder
	.map((label) => ({
		label,
		items: blockCatalog.filter((descriptor) => descriptor.category === label),
	}))
	.filter((group) => group.items.length > 0);

export function matchBlockKindQuery(
	descriptor: BlockKindDescriptor,
	query: string,
): boolean {
	const normalized = query.trim().toLowerCase();

	if (!normalized) {
		return true;
	}

	if (descriptor.label.toLowerCase().includes(normalized)) {
		return true;
	}

	if (descriptor.hint.toLowerCase().includes(normalized)) {
		return true;
	}

	return descriptor.keywords.some((keyword) =>
		keyword.toLowerCase().includes(normalized),
	);
}

export function matchMarkdownShortcut(
	text: string,
): BlockKindDescriptor | null {
	for (const descriptor of blockCatalog) {
		if (descriptor.shortcut && text === descriptor.shortcut) {
			return descriptor;
		}
	}

	return null;
}
