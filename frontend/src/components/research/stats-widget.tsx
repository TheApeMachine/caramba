"use client";

import { useUser } from "@clerk/tanstack-react-start";
import { useLiveQuery } from "@tanstack/react-db";
import { ClientOnly } from "@tanstack/react-router";
import { useMemo } from "react";
import { researchProjectCollection } from "#/collections/research_project";
import { Flex } from "#/components/ui/flex";
import { Loadable, LoadablePending } from "#/components/ui/loadable";
import { Typography } from "#/components/ui/typography";
import { kanbanCardsCollection } from "#/lib/kanban-cards-collection";
import { parseKanbanAssignees } from "#/lib/parse-kanban-assignees";

const StatTile = ({
	label,
	value,
	hint,
}: {
	label: string;
	value: number | string;
	hint?: string;
}) => (
	<Flex.Column justify="between" padding={3} fullHeight>
		<Typography.Span
			variant="muted"
			className="text-xs font-medium uppercase tracking-wide"
		>
			{label}
		</Typography.Span>
		<Typography.Title className="text-3xl font-semibold">
			{value}
		</Typography.Title>
		{hint ? (
			<Typography.Span variant="muted" className="text-xs">
				{hint}
			</Typography.Span>
		) : null}
	</Flex.Column>
);

const StatsGrid = () => {
	const { user } = useUser();
	const projectsQuery = useLiveQuery((query) =>
		query.from({ project: researchProjectCollection }),
	);
	const cardsQuery = useLiveQuery((query) =>
		query.from({ card: kanbanCardsCollection }),
	);

	const isLoading = projectsQuery.isLoading || cardsQuery.isLoading;
	const isError = projectsQuery.isError || cardsQuery.isError;
	const cards = cardsQuery.data ?? [];
	const projectCount = projectsQuery.data?.length ?? 0;
	const openCards = cards.filter((card) => card.column_key !== "done").length;

	const myOpen = useMemo(() => {
		if (!user) {
			return 0;
		}

		return cards.filter(
			(card) =>
				card.column_key !== "done" &&
				parseKanbanAssignees(card.assignees_json).includes(user.id),
		).length;
	}, [cards, user]);

	const overdue = useMemo(() => {
		const now = Date.now();

		return cards.filter(
			(card) =>
				card.column_key !== "done" &&
				card.due_date &&
				card.due_date.getTime() < now,
		).length;
	}, [cards]);

	return (
		<Loadable name="research stats" isLoading={isLoading} isError={isError}>
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
		</Loadable>
	);
};

export const ResearchStatsWidget = () => (
	<ClientOnly fallback={<LoadablePending name="research stats" />}>
		<StatsGrid />
	</ClientOnly>
);
