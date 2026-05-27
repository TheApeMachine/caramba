"use client";

import { useLiveQuery } from "@tanstack/react-db";
import { useAll } from "jazz-tools/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { app } from "../../../schema";
import { researchProjectCollection } from "#/collections/research_project";
import { KanbanColumnView } from "#/components/kanban/column";
import { BoardContext } from "#/components/kanban/context";
import type { KanbanBoard as KanbanBoardState } from "#/components/kanban/model";
import { type BoardAction, boardReducer } from "#/components/kanban/reducer";
import { Flex } from "#/components/ui/flex";
import { Loadable } from "#/components/ui/loadable";
import { ScrollArea } from "#/components/ui/scroll-area";
import { Typography } from "#/components/ui/typography";
import {
	collectOrderingUpdates,
	kanbanBoardFromRows,
} from "#/lib/kanban-board-from-rows";
import { kanbanColumnKeySchema } from "#/lib/kanban-card-schema";
import { useJazzDb } from "#/lib/jazz-db";

export type KanbanBoardScope =
	| { kind: "project"; researchProjectId: string }
	| { kind: "aggregate"; organizationSlug: string };

/*
KanbanBoard renders columns and cards synced through Jazz (the CRDT-primary
store). Reads come from useAll(app.kanbanCards...) and mutations are local-first
writes via db.insert/update/delete — there is no server round-trip or txid await.
The optimistic board state + reducer are preserved so the UI updates instantly;
the pending guard holds the optimistic board until the local durable write lands
and useAll re-emits, after which the synced board takes over.
*/
export function KanbanBoard({ scope }: { scope: KanbanBoardScope }) {
	const db = useJazzDb();
	const pendingMutationRef = useRef(0);

	const cardsQuery = useMemo(
		() =>
			scope.kind === "project"
				? app.kanbanCards.where({ project: scope.researchProjectId })
				: app.kanbanCards.where({ organization_slug: scope.organizationSlug }),
		[scope],
	);

	const cardRows = useAll(cardsQuery);
	const cardsLoading = cardRows === undefined;

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

	const filteredCardRows = useMemo(() => cardRows ?? [], [cardRows]);

	const projectsById = useMemo(() => {
		const map = new Map<string, { name: string }>();

		for (const project of projectsQuery.data ?? []) {
			map.set(project.id, { name: project.name });
		}

		return map;
	}, [projectsQuery.data]);

	const projectOrgSlugById = useMemo(() => {
		const map = new Map<string, string>();

		for (const project of projectsQuery.data ?? []) {
			map.set(project.id, project.organization_slug ?? "");
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
					const column = previous.columns.find(
						(entry) => entry.id === action.columnId,
					);
					const sortOrder = column?.cardIds.length ?? 0;
					const dueDate =
						action.card.dueDate !== null && action.card.dueDate !== ""
							? new Date(`${action.card.dueDate}T12:00:00Z`)
							: null;

					const inserted = db.insert(app.kanbanCards, {
						project: scope.researchProjectId,
						organization_slug:
							projectOrgSlugById.get(scope.researchProjectId) ?? "",
						column_key: kanbanColumnKeySchema.parse(action.columnId),
						sort_order: sortOrder,
						title: action.card.title.trim(),
						description: action.card.description.trim(),
						priority: action.card.priority,
						labels: action.card.labels,
						assignees: action.card.assignees,
						due_date: dueDate,
						created_at: new Date(),
						updated_at: new Date(),
					});

					void inserted.wait({ tier: "local" }).finally(() => {
						pendingMutationRef.current--;
					});

					return boardReducer(previous, {
						...action,
						preferredCardId: inserted.value.id,
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

					pendingMutationRef.current++;

					const now = new Date();
					const handles = collectOrderingUpdates(nextBoard).map((update) =>
						db.update(app.kanbanCards, update.id, {
							column_key: update.column_key,
							sort_order: update.sort_order,
							updated_at: now,
						}),
					);

					void Promise.allSettled(
						handles.map((handle) => handle.wait({ tier: "local" })),
					).finally(() => {
						pendingMutationRef.current--;
					});

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

					pendingMutationRef.current++;

					const dueDate =
						updatedCard.dueDate !== null && updatedCard.dueDate !== ""
							? new Date(`${updatedCard.dueDate}T12:00:00Z`)
							: null;

					const handle = db.update(app.kanbanCards, action.id, {
						title: updatedCard.title,
						description: updatedCard.description,
						priority: updatedCard.priority,
						labels: updatedCard.labels,
						assignees: updatedCard.assignees,
						due_date: dueDate,
						updated_at: new Date(),
					});

					void handle.wait({ tier: "local" }).finally(() => {
						pendingMutationRef.current--;
					});

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

					pendingMutationRef.current++;

					const handle = db.delete(app.kanbanCards, action.id);

					void handle.wait({ tier: "local" }).finally(() => {
						pendingMutationRef.current--;
					});

					return nextBoard;
				});

				return;
			}

			setBoard((previous) => boardReducer(previous, action));
		},
		[scope, db, projectOrgSlugById],
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

	const kanbanError = (
		<Flex.Center padding={6} className="flex-1 text-center">
			<Typography.Paragraph variant="muted">
				Could not load Kanban projects. Confirm the research projects shape is
				reachable.
			</Typography.Paragraph>
		</Flex.Center>
	);

	return (
		<Loadable
			name="board"
			isLoading={cardsLoading || projectsQuery.isLoading}
			isError={projectsQuery.isError}
			error={kanbanError}
		>
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
		</Loadable>
	);
}
