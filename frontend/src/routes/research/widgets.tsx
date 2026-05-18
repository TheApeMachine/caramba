"use client";

import { useUser } from "@clerk/tanstack-react-start";
import { useLiveQuery } from "@tanstack/react-db";
import { Link } from "@tanstack/react-router";
import {
	CalendarIcon,
	FolderIcon,
	ListChecksIcon,
	PlusIcon,
} from "lucide-react";
import { useMemo } from "react";
import { researchProjectCollection } from "#/collections/research_project";
import type { WidgetDescriptor } from "#/components/dashboard/registry";
import { PRIORITY_COLORS, type Priority } from "#/components/kanban/model";
import { QuickActionsWidget } from "#/components/research/quick-actions-widget";
import { ResearcherProfileWidget } from "#/components/research/researcher-profile-widget";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import { kanbanCardsCollection } from "#/lib/kanban-cards-collection";

/*
parseAssignees mirrors lib/kanban-board-from-rows.ts but returns just the
list of assignee identifiers — the only thing the dashboard widgets need.
*/
const parseAssignees = (raw: string): string[] => {
	try {
		const value: unknown = JSON.parse(raw);
		if (!Array.isArray(value)) return [];
		return value.filter((entry): entry is string => typeof entry === "string");
	} catch {
		return [];
	}
};

const StatTile = ({
	label,
	value,
	hint,
}: {
	label: string;
	value: number | string;
	hint?: string;
}) => (
	<div className="flex h-full flex-col justify-between p-3">
		<div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
			{label}
		</div>
		<div className="font-semibold text-3xl text-foreground">{value}</div>
		{hint && <div className="text-xs text-muted-foreground">{hint}</div>}
	</div>
);

const Empty = ({ children }: { children: React.ReactNode }) => (
	<div className="flex h-full items-center justify-center px-3 py-6 text-center text-muted-foreground text-xs">
		{children}
	</div>
);

const ProjectsWidget = () => {
	const { data, isLoading } = useLiveQuery((q) =>
		q.from({ project: researchProjectCollection }),
	);
	const projects = data ?? [];

	if (isLoading) return <Empty>Loading projects…</Empty>;
	if (projects.length === 0)
		return (
			<div className="flex h-full flex-col items-center justify-center gap-3 p-4 text-center">
				<p className="text-muted-foreground text-sm">No projects yet.</p>
				<Button render={<Link to="/research/new" />} size="sm">
					<PlusIcon /> New project
				</Button>
			</div>
		);

	return (
		<ul className="flex flex-col gap-1 p-1">
			{projects.map((project) => (
				<li key={project.id}>
					<Button
						className="h-auto w-full justify-start py-2"
						render={
							<Link to="/research/edit" search={{ projectId: project.id }} />
						}
						variant="ghost"
					>
						<FolderIcon className="opacity-60" />
						<div className="flex flex-col items-start gap-0.5 truncate">
							<span className="truncate font-medium text-sm">
								{project.name}
							</span>
							{project.description && (
								<span className="truncate font-normal text-muted-foreground text-xs">
									{project.description}
								</span>
							)}
						</div>
					</Button>
				</li>
			))}
		</ul>
	);
};

const MyTodosWidget = () => {
	const { user, isLoaded } = useUser();
	const { data, isLoading } = useLiveQuery((q) =>
		q.from({ card: kanbanCardsCollection }),
	);

	const projects = useLiveQuery((q) =>
		q.from({ project: researchProjectCollection }),
	).data;
	const projectName = useMemo(() => {
		const map = new Map<string, string>();
		for (const project of projects ?? []) map.set(project.id, project.name);
		return map;
	}, [projects]);

	const mine = useMemo(() => {
		if (!user || !data) return [];
		const me = user.id;
		return data
			.filter((row) => row.column_key !== "done")
			.filter((row) => parseAssignees(row.assignees_json).includes(me))
			.sort((a, b) => {
				const ad = a.due_date ? a.due_date.getTime() : Number.POSITIVE_INFINITY;
				const bd = b.due_date ? b.due_date.getTime() : Number.POSITIVE_INFINITY;
				return ad - bd;
			});
	}, [data, user]);

	if (!isLoaded || isLoading) return <Empty>Loading todos…</Empty>;
	if (!user) return <Empty>Sign in to see your todos.</Empty>;
	if (mine.length === 0)
		return <Empty>Nothing assigned to you — enjoy the quiet.</Empty>;

	return (
		<ul className="flex flex-col gap-1 p-1">
			{mine.slice(0, 12).map((card) => {
				const overdue = card.due_date && card.due_date < new Date();
				return (
					<li
						key={card.id}
						className="flex items-start gap-2 rounded-md px-2 py-1.5 hover:bg-muted/40"
					>
						<Badge
							size="sm"
							variant={PRIORITY_COLORS[card.priority as Priority]}
						>
							{card.priority}
						</Badge>
						<div className="flex min-w-0 flex-1 flex-col">
							<span className="truncate font-medium text-sm">{card.title}</span>
							<span className="flex items-center gap-2 text-muted-foreground text-xs">
								{projectName.get(card.research_project_id) ?? "—"}
								{card.due_date && (
									<span
										className={
											overdue
												? "flex items-center gap-1 text-destructive-foreground"
												: "flex items-center gap-1"
										}
									>
										<CalendarIcon className="size-3" />
										{card.due_date.toLocaleDateString()}
									</span>
								)}
							</span>
						</div>
					</li>
				);
			})}
		</ul>
	);
};

const RecentActivityWidget = () => {
	const { data: cards } = useLiveQuery((q) =>
		q.from({ card: kanbanCardsCollection }),
	);
	const { data: projects } = useLiveQuery((q) =>
		q.from({ project: researchProjectCollection }),
	);

	const items = useMemo(() => {
		const out: Array<{
			id: string;
			title: string;
			when: Date;
			kind: "project" | "card";
			sub?: string;
		}> = [];
		for (const project of projects ?? [])
			out.push({
				id: `p:${project.id}`,
				title: project.name,
				when: project.updated_at,
				kind: "project",
			});
		for (const card of cards ?? [])
			out.push({
				id: `c:${card.id}`,
				title: card.title,
				when: card.updated_at,
				kind: "card",
				sub: card.column_key,
			});
		return out.sort((a, b) => b.when.getTime() - a.when.getTime()).slice(0, 10);
	}, [cards, projects]);

	if (items.length === 0) return <Empty>No recent activity.</Empty>;

	return (
		<ul className="flex flex-col gap-1 p-1">
			{items.map((item) => (
				<li
					key={item.id}
					className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-muted/40"
				>
					{item.kind === "project" ? (
						<FolderIcon className="size-3.5 opacity-60" />
					) : (
						<ListChecksIcon className="size-3.5 opacity-60" />
					)}
					<span className="flex-1 truncate">{item.title}</span>
					{item.sub && (
						<Badge size="sm" variant="outline">
							{item.sub}
						</Badge>
					)}
					<span className="text-muted-foreground text-xs">
						{item.when.toLocaleDateString()}
					</span>
				</li>
			))}
		</ul>
	);
};

const StatsWidget = () => {
	const { user } = useUser();
	const { data: projects } = useLiveQuery((q) =>
		q.from({ project: researchProjectCollection }),
	);
	const { data: cards } = useLiveQuery((q) =>
		q.from({ card: kanbanCardsCollection }),
	);

	const projectCount = projects?.length ?? 0;
	const openCards =
		cards?.filter((card) => card.column_key !== "done").length ?? 0;
	const myOpen = useMemo(() => {
		if (!user || !cards) return 0;
		return cards.filter(
			(card) =>
				card.column_key !== "done" &&
				parseAssignees(card.assignees_json).includes(user.id),
		).length;
	}, [cards, user]);
	const overdue = useMemo(() => {
		if (!cards) return 0;
		const now = new Date();
		return cards.filter(
			(card) =>
				card.column_key !== "done" && card.due_date && card.due_date < now,
		).length;
	}, [cards]);

	return (
		<div className="grid h-full grid-cols-2 gap-2">
			<StatTile label="Projects" value={projectCount} />
			<StatTile label="Open tasks" value={openCards} />
			<StatTile label="Assigned to me" value={myOpen} />
			<StatTile
				label="Overdue"
				value={overdue}
				hint={overdue > 0 ? "needs attention" : undefined}
			/>
		</div>
	);
};

export const researchWidgets: WidgetDescriptor[] = [
	{
		kind: "research-actions",
		title: "",
		description: "",
		hideHeader: true,
		render: () => <QuickActionsWidget />,
	},
	{
		kind: "research-stats",
		title: "At a glance",
		description: "Project and task counts",
		render: () => <StatsWidget />,
	},
	{
		kind: "research-projects",
		title: "Projects",
		description: "Your research projects",
		render: () => <ProjectsWidget />,
	},
	{
		kind: "research-todos",
		title: "My todos",
		description: "Kanban cards assigned to you",
		render: () => <MyTodosWidget />,
	},
	{
		kind: "research-activity",
		title: "Recent activity",
		description: "Latest updates across the workspace",
		render: () => <RecentActivityWidget />,
	},
	{
		kind: "research-profile",
		title: "",
		description: "",
		hideHeader: true,
		render: () => <ResearcherProfileWidget />,
	},
];

/*
defaultResearchLayout pre-arranges widgets so the page is useful before
the user touches anything. The grid is 4 cols x 2 rows.
*/
export const defaultResearchLayout = [
	{
		id: "r-actions",
		kind: "research-actions",
		col: 0,
		row: 0,
		colSpan: 1,
		rowSpan: 2,
	},
	{
		id: "r-stats",
		kind: "research-stats",
		col: 1,
		row: 0,
		colSpan: 2,
		rowSpan: 1,
	},
	{
		id: "r-profile",
		kind: "research-profile",
		col: 3,
		row: 0,
		colSpan: 1,
		rowSpan: 1,
	},
	{
		id: "r-projects",
		kind: "research-projects",
		col: 1,
		row: 1,
		colSpan: 1,
		rowSpan: 1,
	},
	{
		id: "r-todos",
		kind: "research-todos",
		col: 2,
		row: 1,
		colSpan: 2,
		rowSpan: 1,
	},
];
