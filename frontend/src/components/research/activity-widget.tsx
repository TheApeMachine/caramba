"use client";

import { useLiveQuery } from "@tanstack/react-db";
import { FolderIcon, ListChecksIcon } from "lucide-react";
import { useMemo } from "react";
import { researchProjectCollection } from "#/collections/research_project";
import { Component } from "#/components/component";
import { Badge } from "#/components/ui/badge";
import { kanbanCardsCollection } from "#/lib/kanban-cards-collection";

type ResearchProjectActivityItem = {
	id: string;
	name: string;
	updated_at: Date;
};

const formatActivityDay = (date: Date) => date.toISOString().slice(0, 10);

const ActivityList = ({
	projects,
}: {
	projects: ResearchProjectActivityItem[];
}) => {
	const { data: cards } = useLiveQuery((query) =>
		query.from({ card: kanbanCardsCollection }),
	);

	const items = useMemo(() => {
		const entries: Array<{
			id: string;
			title: string;
			when: Date;
			kind: "project" | "card";
			sub?: string;
		}> = [];

		for (const project of projects) {
			entries.push({
				id: `p:${project.id}`,
				title: project.name,
				when: project.updated_at,
				kind: "project",
			});
		}

		for (const card of cards ?? []) {
			entries.push({
				id: `c:${card.id}`,
				title: card.title,
				when: card.updated_at,
				kind: "card",
				sub: card.column_key,
			});
		}

		return entries
			.sort((left, right) => right.when.getTime() - left.when.getTime())
			.slice(0, 10);
	}, [cards, projects]);

	if (items.length === 0) {
		return (
			<div className="flex h-full items-center justify-center px-3 py-6 text-center text-muted-foreground text-xs">
				No recent activity.
			</div>
		);
	}

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
					{item.sub ? (
						<Badge size="sm" variant="outline">
							{item.sub}
						</Badge>
					) : null}
					<span className="text-muted-foreground text-xs">
						{formatActivityDay(item.when)}
					</span>
				</li>
			))}
		</ul>
	);
};

export const ResearchActivityWidget = () => (
	<Component<ResearchProjectActivityItem[]>
		name="recent activity"
		isEmpty={() => false}
		query={(query) =>
			query
				.from({ project: researchProjectCollection })
				.select(({ project }) => ({
					id: project.id,
					name: project.name,
					updated_at: project.updated_at,
				}))
		}
	>
		{(projects) => <ActivityList projects={projects} />}
	</Component>
);
