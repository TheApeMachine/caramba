import type { KanbanBoard, KanbanCard, KanbanColumn } from "./model";

export type BoardAction =
	| { type: "ADD_COLUMN"; title: string; color: string }
	| {
			type: "UPDATE_COLUMN";
			id: string;
			title?: string;
			color?: string;
			limit?: number | null;
	  }
	| { type: "DELETE_COLUMN"; id: string }
	| { type: "REORDER_COLUMNS"; columnIds: string[] }
	| {
			type: "ADD_CARD";
			columnId: string;
			preferredCardId?: string;
			card: Omit<KanbanCard, "id" | "createdAt" | "order" | "columnId">;
	  }
	| {
			type: "UPDATE_CARD";
			id: string;
			changes: Partial<Omit<KanbanCard, "id" | "createdAt" | "columnId">>;
	  }
	| { type: "DELETE_CARD"; id: string }
	| {
			type: "MOVE_CARD";
			cardId: string;
			fromColumnId: string;
			toColumnId: string;
			toIndex: number;
	  }
	| { type: "REORDER_CARDS"; columnId: string; cardIds: string[] };

let nextId = 1;

const uid = () => `${Date.now()}-${nextId++}`;

export function boardReducer(
	state: KanbanBoard,
	action: BoardAction,
): KanbanBoard {
	switch (action.type) {
		case "ADD_COLUMN": {
			const id = uid();
			return {
				...state,
				columns: [
					...state.columns,
					{
						id,
						title: action.title,
						color: action.color,
						cardIds: [],
						limit: null,
					},
				],
			};
		}

		case "UPDATE_COLUMN": {
			return {
				...state,
				columns: state.columns.map((col) =>
					col.id !== action.id
						? col
						: {
								...col,
								...(action.title !== undefined && { title: action.title }),
								...(action.color !== undefined && { color: action.color }),
								...(action.limit !== undefined && { limit: action.limit }),
							},
				),
			};
		}

		case "DELETE_COLUMN": {
			const column = state.columns.find((col) => col.id === action.id);
			if (!column) return state;
			const updatedCards = { ...state.cards };
			column.cardIds.forEach((cardId) => {
				delete updatedCards[cardId];
			});
			return {
				columns: state.columns.filter((col) => col.id !== action.id),
				cards: updatedCards,
			};
		}

		case "REORDER_COLUMNS": {
			const colMap = new Map(state.columns.map((col) => [col.id, col]));
			return {
				...state,
				columns: action.columnIds
					.map((id) => colMap.get(id))
					.filter((column): column is KanbanColumn => column !== undefined),
			};
		}

		case "ADD_CARD": {
			const id = action.preferredCardId ?? uid();
			const column = state.columns.find((col) => col.id === action.columnId);
			if (!column) return state;
			const newCard: KanbanCard = {
				...action.card,
				id,
				columnId: action.columnId,
				order: column.cardIds.length,
				createdAt: new Date().toISOString(),
			};
			return {
				cards: { ...state.cards, [id]: newCard },
				columns: state.columns.map((col) =>
					col.id !== action.columnId
						? col
						: { ...col, cardIds: [...col.cardIds, id] },
				),
			};
		}

		case "UPDATE_CARD": {
			const card = state.cards[action.id];
			if (!card) return state;
			return {
				...state,
				cards: { ...state.cards, [action.id]: { ...card, ...action.changes } },
			};
		}

		case "DELETE_CARD": {
			const card = state.cards[action.id];
			if (!card) return state;
			const updatedCards = { ...state.cards };
			delete updatedCards[action.id];
			return {
				cards: updatedCards,
				columns: state.columns.map((col) =>
					col.id !== card.columnId
						? col
						: {
								...col,
								cardIds: col.cardIds.filter((id) => id !== action.id),
							},
				),
			};
		}

		case "MOVE_CARD": {
			const { cardId, fromColumnId, toColumnId, toIndex } = action;
			const card = state.cards[cardId];
			if (!card) return state;

			const columns = state.columns.map((col) => {
				if (col.id === fromColumnId && col.id !== toColumnId) {
					return { ...col, cardIds: col.cardIds.filter((id) => id !== cardId) };
				}
				if (col.id === toColumnId && col.id !== fromColumnId) {
					const next = [...col.cardIds];
					next.splice(toIndex, 0, cardId);
					return { ...col, cardIds: next };
				}
				if (col.id === fromColumnId && col.id === toColumnId) {
					const next = col.cardIds.filter((id) => id !== cardId);
					next.splice(toIndex, 0, cardId);
					return { ...col, cardIds: next };
				}
				return col;
			});

			return {
				columns,
				cards: { ...state.cards, [cardId]: { ...card, columnId: toColumnId } },
			};
		}

		case "REORDER_CARDS": {
			return {
				...state,
				columns: state.columns.map((col) =>
					col.id !== action.columnId
						? col
						: { ...col, cardIds: action.cardIds },
				),
			};
		}

		default:
			return state;
	}
}
