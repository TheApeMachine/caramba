"use client";

import { useLiveQuery } from "@tanstack/react-db";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { researchProjectCollection } from "#/collections/research_project";
import { KanbanColumnView } from "#/components/kanban/column";
import { BoardContext } from "#/components/kanban/context";
import type { KanbanBoard as KanbanBoardState } from "#/components/kanban/model";
import { type BoardAction, boardReducer } from "#/components/kanban/reducer";
import { Flex } from "#/components/ui/flex";
import { ScrollArea } from "#/components/ui/scroll-area";
import { Typography } from "#/components/ui/typography";
import {
	assigneesJsonFromKanban,
	collectOrderingUpdates,
	kanbanBoardFromRows,
	labelsJsonFromKanban,
} from "#/lib/kanban-board-from-rows";
import { kanbanColumnKeySchema } from "#/lib/kanban-card-schema";
import { kanbanCardsCollection } from "#/lib/kanban-cards-collection";
import {
	deleteKanbanCard,
	patchKanbanCard,
	syncKanbanOrdering,
} from "#/server/kanban-cards";

export type KanbanBoardScope =
	| { kind: "project"; researchProjectId: string }
	| { kind: "aggregate"; organizationSlug: string };

/*
KanbanBoard renders columns and cards synced via Electric and persists mutations through Postgres.
*/
export function KanbanBoard({ scope }: { scope: KanbanBoardScope }) {
	const pendingMutationRef = useRef(0);
	const cardsQuery = useLiveQuery((query) =>
		query.from({ card: kanbanCardsCollection }).select(({ card }) => ({
			id: card.id,
			research_project_id: card.research_project_id,
			column_key: card.column_key,
			sort_order: card.sort_order,
			title: card.title,
			description: card.description,
			priority: card.priority,
			labels_json: card.labels_json,
			assignees_json: card.assignees_json,
			due_date: card.due_date,
			requested_by: card.requested_by,
			created_at: card.created_at,
			updated_at: card.updated_at,
		})),
	);

	const projectsQuery = useLiveQuery((query) =>
		query
			.from({ project: researchProjectCollection })
			.select(({ project }) => ({
				id: project.id,
				name: project.name,
				organization_slug: project.organization_slug,
				project_slug: project.project_slug,
			})),
	);

	const filteredCardRows = useMemo(() => {
		const rows = cardsQuery.data ?? [];

		if (scope.kind === "project") {
			return rows.filter(
				(row) => row.research_project_id === scope.researchProjectId,
			);
		}

		const organizationProjectIdentifiers = new Set(
			(projectsQuery.data ?? [])
				.filter(
					(project) => project.organization_slug === scope.organizationSlug,
				)
				.map((project) => project.id),
		);

		return rows.filter((row) =>
			organizationProjectIdentifiers.has(row.research_project_id),
		);
	}, [cardsQuery.data, projectsQuery.data, scope]);

	const projectsById = useMemo(() => {
		const map = new Map<string, { name: string }>();

		for (const project of projectsQuery.data ?? []) {
			map.set(project.id, { name: project.name });
		}

		return map;
	}, [projectsQuery.data]);

	const syncedBoard = useMemo(
		() =>
			kanbanBoardFromRows(
				filteredCardRows,
				projectsById,
				scope.kind === "aggregate",
			),
		[filteredCardRows, projectsById, scope.kind],
	);

	const [board, setBoard] = useState<KanbanBoardState>(syncedBoard);

	const [dragState, setDragState] = useState<{
		cardId: string;
		fromColumnId: string;
	} | null>(null);

	useEffect(() => {
		if (pendingMutationRef.current > 0) {
			return;
		}

		setBoard(syncedBoard);
	}, [syncedBoard]);

	const wrappedDispatch = useCallback(
		(action: BoardAction) => {
			if (
				action.type === "DELETE_COLUMN" ||
				action.type === "UPDATE_COLUMN" ||
				action.type === "REORDER_COLUMNS"
			) {
				return;
			}

			if (action.type === "ADD_CARD") {
				if (scope.kind !== "project") {
					return;
				}

				pendingMutationRef.current++;

				setBoard((previous) => {
					const preferredCardId = action.preferredCardId ?? crypto.randomUUID();
					const column = previous.columns.find(
						(entry) => entry.id === action.columnId,
					);
					const sortOrder = column?.cardIds.length ?? 0;

					const transaction = kanbanCardsCollection.insert({
						id: preferredCardId,
						research_project_id: scope.researchProjectId,
						column_key: kanbanColumnKeySchema.parse(action.columnId),
						sort_order: sortOrder,
						title: action.card.title.trim(),
						description: action.card.description.trim(),
						priority: action.card.priority,
						labels_json: labelsJsonFromKanban(action.card.labels),
						assignees_json: assigneesJsonFromKanban(action.card.assignees),
						due_date:
							action.card.dueDate !== null && action.card.dueDate !== ""
								? new Date(`${action.card.dueDate}T12:00:00Z`)
								: null,
						requested_by: null,
						created_at: new Date(),
						updated_at: new Date(),
					});

					void transaction.isPersisted.promise.finally(() => {
						pendingMutationRef.current--;
					});

					return boardReducer(previous, {
						...action,
						preferredCardId,
					});
				});

				return;
			}

			if (action.type === "MOVE_CARD") {
				setBoard((previous) => {
					const nextBoard = boardReducer(previous, action);

					if (nextBoard === previous) {
						return previous;
					}

					void (async () => {
						pendingMutationRef.current++;

						try {
							const result = await syncKanbanOrdering({
								data: {
									updates: collectOrderingUpdates(nextBoard),
								},
							});

							await kanbanCardsCollection.utils.awaitTxId(result.txid, 60_000);
						} finally {
							pendingMutationRef.current--;
						}
					})();

					return nextBoard;
				});

				return;
			}

			if (action.type === "UPDATE_CARD") {
				setBoard((previous) => {
					const nextBoard = boardReducer(previous, action);
					const updatedCard = nextBoard.cards[action.id];

					if (!updatedCard || nextBoard === previous) {
						return previous;
					}

					void (async () => {
						pendingMutationRef.current++;

						try {
							const result = await patchKanbanCard({
								data: {
									id: action.id,
									title: updatedCard.title,
									description: updatedCard.description,
									priority: updatedCard.priority,
									labels_json: labelsJsonFromKanban(updatedCard.labels),
									assignees_json: assigneesJsonFromKanban(
										updatedCard.assignees,
									),
									due_date:
										updatedCard.dueDate !== null && updatedCard.dueDate !== ""
											? new Date(`${updatedCard.dueDate}T12:00:00Z`)
											: null,
								},
							});

							await kanbanCardsCollection.utils.awaitTxId(result.txid, 60_000);
						} finally {
							pendingMutationRef.current--;
						}
					})();

					return nextBoard;
				});

				return;
			}

			if (action.type === "DELETE_CARD") {
				setBoard((previous) => {
					const nextBoard = boardReducer(previous, action);

					if (nextBoard === previous) {
						return previous;
					}

					void (async () => {
						pendingMutationRef.current++;

						try {
							const result = await deleteKanbanCard({
								data: { id: action.id },
							});

							await kanbanCardsCollection.utils.awaitTxId(result.txid, 60_000);
						} finally {
							pendingMutationRef.current--;
						}
					})();

					return nextBoard;
				});

				return;
			}

			setBoard((previous) => boardReducer(previous, action));
		},
		[scope],
	);

	const handleDragStart = (e: React.DragEvent) => {
		const cardId = e.dataTransfer.getData("cardId");
		const fromColumnId = e.dataTransfer.getData("fromColumnId");

		if (cardId && fromColumnId) {
			setDragState({ cardId, fromColumnId });
		}
	};

	const handleDragOver = (
		e: React.DragEvent,
		_columnId: string,
		_index: number,
	) => {
		e.preventDefault();
	};

	const handleDrop = (
		e: React.DragEvent,
		toColumnId: string,
		toIndex: number,
	) => {
		e.preventDefault();
		const cardId = e.dataTransfer.getData("cardId");
		const fromColumnId = e.dataTransfer.getData("fromColumnId");

		if (!cardId || !fromColumnId) {
			return;
		}

		wrappedDispatch({
			type: "MOVE_CARD",
			cardId,
			fromColumnId,
			toColumnId,
			toIndex,
		});
		setDragState(null);
	};

	const columnsEditable = false;
	const allowAddCard = scope.kind === "project";

	if (cardsQuery.isLoading || projectsQuery.isLoading) {
		return (
			<Flex.Center className="flex-1 p-6">
				<Typography.Paragraph variant="muted">
					Loading board…
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	if (cardsQuery.isError || projectsQuery.isError) {
		return (
			<Flex.Center className="flex-1 p-6 text-center">
				<Typography.Paragraph variant="muted">
					Could not sync Kanban data. Confirm{" "}
					<Typography.Code>VITE_ELECTRIC_SHAPE_URL</Typography.Code> exposes
					shapes for <Typography.Code>kanban_cards</Typography.Code> and{" "}
					<Typography.Code>research_projects</Typography.Code>, and{" "}
					<Typography.Code>DATABASE_URL</Typography.Code> is set for server
					writes.
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	return (
		<BoardContext.Provider value={{ board, dispatch: wrappedDispatch }}>
			<Flex.Column className="min-h-0 flex-1">
				<ScrollArea className="min-h-0 flex-1">
					<ul
						className="flex h-full min-h-0 list-none gap-3 px-1 py-1"
						onDragEnd={() => setDragState(null)}
						onDragStart={handleDragStart}
					>
						{board.columns.map((column) => (
							<li className="flex min-w-72 flex-1" key={column.id}>
								<KanbanColumnView
									allowAddCard={allowAddCard}
									column={column}
									columnsEditable={columnsEditable}
									dragState={dragState}
									onDragOver={handleDragOver}
									onDrop={handleDrop}
								/>
							</li>
						))}
					</ul>
				</ScrollArea>
			</Flex.Column>
		</BoardContext.Provider>
	);
}
