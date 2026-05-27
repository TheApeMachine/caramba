export const NEW_PROJECT_STARTER_CARDS = [
	{
		title: "Project kickoff",
		description:
			"Align on goals, scope, and success criteria for this research effort.",
		columnKey: "todo" as const,
	},
	{
		title: "Draft architecture",
		description: "Sketch the model graph, data flow, and planned ablations.",
		columnKey: "backlog" as const,
	},
	{
		title: "Paper outlines",
		description:
			"Draft one or more papers tied to this project as results mature.",
		columnKey: "backlog" as const,
	},
] as const;

export const MAX_PROJECT_PAPERS_AT_PROVISION = 20;

export type NewResearchProjectPaperDraft = {
	id: string;
	title: string;
};

export type NewResearchProjectSpec = {
	id: string;
	name: string;
	description: string;
	projectSlug: string;
	memberIds: string[];
	papers: NewResearchProjectPaperDraft[];
};

export const createPaperDraft = (title = ""): NewResearchProjectPaperDraft => ({
	id: crypto.randomUUID(),
	title,
});

export const emptyNewResearchProjectSpec = (): NewResearchProjectSpec => ({
	id: crypto.randomUUID(),
	name: "",
	description: "",
	projectSlug: "",
	memberIds: [],
	papers: [createPaperDraft()],
});
