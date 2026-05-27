"use client";

import { useLiveQuery } from "@tanstack/react-db";
import { ClientOnly } from "@tanstack/react-router";
import { FolderIcon, ListChecksIcon } from "lucide-react";
import { useMemo } from "react";
import { researchProjectCollection } from "#/collections/research_project";
import { Badge } from "#/components/ui/badge";
import { Flex } from "#/components/ui/flex";
import { Loadable, LoadablePending } from "#/components/ui/loadable";
import { Typography } from "#/components/ui/typography";
import { kanbanCardsCollection } from "#/lib/kanban-cards-collection";

type ActivityEntry = {
	id: string;
	title: string;
	when: Date;
	kind: "project" | "card";
	sub?: string;
};

const formatActivityDay = (date: Date) => date.toISOString().slice(0, 10);

const ActivityRow = ({ entry }: { entry: ActivityEntry }) => (
	<Flex.Row
		align="center"
		gap={2}
		className="rounded-md px-2 py-1.5 text-sm hover:bg-muted/40"
	>
		{entry.kind === "project" ? (
			<FolderIcon className="size-3.5 opacity-60" />
		) : (
			<ListChecksIcon className="size-3.5 opacity-60" />
		)}
		<Typography.Span truncate className="flex-1 text-sm">
			{entry.title}
		</Typography.Span>
		{entry.sub ? (
			<Badge size="sm" variant="outline">
				{entry.sub}
			</Badge>
		) : null}
		<Typography.Span variant="muted" className="text-xs">
			{formatActivityDay(entry.when)}
		</Typography.Span>
	</Flex.Row>
);

const ActivityEmpty = () => (
	<Flex.Center fullHeight padding={4}>
		<Typography.Paragraph variant="muted">No recent activity.</Typography.Paragraph>
	</Flex.Center>
);

const ActivityFeed = () => {
	const projectsQuery = useLiveQuery((query) =>
		query.from({ project: researchProjectCollection }).select(({ project }) => ({
			id: project.id,
			name: project.name,
			updated_at: project.updated_at,
		})),
	);

	const cardsQuery = useLiveQuery((query) =>
		query.from({ card: kanbanCardsCollection }),
	);

	const isLoading = projectsQuery.isLoading || cardsQuery.isLoading;
	const isError = projectsQuery.isError || cardsQuery.isError;

	const items = useMemo<ActivityEntry[]>(() => {
		const entries: ActivityEntry[] = [];

		for (const project of projectsQuery.data ?? []) {
			entries.push({
				id: `p:${project.id}`,
				title: project.name,
				when: project.updated_at,
				kind: "project",
			});
		}

		for (const card of cardsQuery.data ?? []) {
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
	}, [projectsQuery.data, cardsQuery.data]);

	return (
		<Loadable
			name="recent activity"
			isLoading={isLoading}
			isError={isError}
			isEmpty={items.length === 0}
			empty={<ActivityEmpty />}
		>
			<Flex.Column gap={1} padding={1} className="list-none">
				{items.map((entry) => (
					<li key={entry.id}>
						<ActivityRow entry={entry} />
					</li>
				))}
			</Flex.Column>
		</Loadable>
	);
};

export const ResearchActivityWidget = () => (
	<ClientOnly fallback={<LoadablePending name="recent activity" />}>
		<ActivityFeed />
	</ClientOnly>
);
