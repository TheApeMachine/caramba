export type Priority = "low" | "medium" | "high" | "critical";
export type CardLabel = { id: string; text: string; color: string };

export interface KanbanCard {
	id: string;
	title: string;
	description: string;
	priority: Priority;
	labels: CardLabel[];
	assignees: string[];
	dueDate: string | null;
	columnId: string;
	order: number;
	createdAt: string;
	researchProjectId?: string;
	sourceProjectName?: string;
}

export interface KanbanColumn {
	id: string;
	title: string;
	color: string;
	cardIds: string[];
	limit: number | null;
}

export interface KanbanBoard {
	columns: KanbanColumn[];
	cards: Record<string, KanbanCard>;
}

type BadgeVariant = "success" | "info" | "warning" | "destructive";

export const PRIORITY_COLORS: Record<Priority, BadgeVariant> = {
	low: "success",
	medium: "info",
	high: "warning",
	critical: "destructive",
} as const;

export const LABEL_PALETTE = [
	"#ef4444",
	"#f97316",
	"#eab308",
	"#22c55e",
	"#3b82f6",
	"#8b5cf6",
	"#ec4899",
	"#14b8a6",
];

export const DEFAULT_BOARD: KanbanBoard = {
	columns: [
		{
			id: "backlog",
			title: "Backlog",
			color: "#6b7280",
			cardIds: [],
			limit: null,
		},
		{ id: "todo", title: "To Do", color: "#3b82f6", cardIds: [], limit: null },
		{
			id: "in-progress",
			title: "In Progress",
			color: "#f97316",
			cardIds: [],
			limit: 5,
		},
		{ id: "review", title: "Review", color: "#8b5cf6", cardIds: [], limit: 3 },
		{ id: "done", title: "Done", color: "#22c55e", cardIds: [], limit: null },
	],
	cards: {},
};
