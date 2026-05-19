"use client";

import { useUser } from "@clerk/tanstack-react-start";
import { useLiveQuery } from "@tanstack/react-db";
import { useMemo } from "react";
import { researchProjectCollection } from "#/collections/research_project";
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
	<div className="flex h-full flex-col justify-between p-3">
		<div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
			{label}
		</div>
		<div className="font-semibold text-3xl text-foreground">{value}</div>
		{hint ? <div className="text-xs text-muted-foreground">{hint}</div> : null}
	</div>
);

export const ResearchStatsWidget = () => {
	const { user } = useUser();
	const { data: projects } = useLiveQuery((query) =>
		query.from({ project: researchProjectCollection }),
	);
	const { data: cards } = useLiveQuery((query) =>
		query.from({ card: kanbanCardsCollection }),
	);

	const projectCount = projects?.length ?? 0;
	const openCards =
		cards?.filter((card) => card.column_key !== "done").length ?? 0;

	const myOpen = useMemo(() => {
		if (!user || !cards) {
			return 0;
		}

		return cards.filter(
			(card) =>
				card.column_key !== "done" &&
				parseKanbanAssignees(card.assignees_json).includes(user.id),
		).length;
	}, [cards, user]);

	const overdue = useMemo(() => {
		if (!cards) {
			return 0;
		}

		const now = Date.now();

		return cards.filter(
			(card) =>
				card.column_key !== "done" &&
				card.due_date &&
				card.due_date.getTime() < now,
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
