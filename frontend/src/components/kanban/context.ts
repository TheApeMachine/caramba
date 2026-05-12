import { createContext, useContext } from "react";
import type { KanbanBoard } from "./model";
import type { BoardAction } from "./reducer";

interface BoardContextValue {
	board: KanbanBoard;
	dispatch: React.Dispatch<BoardAction>;
}

export const BoardContext = createContext<BoardContextValue | null>(null);

export function useBoardContext(): BoardContextValue {
	const ctx = useContext(BoardContext);
	if (!ctx) throw new Error("useBoardContext must be used inside KanbanBoard");
	return ctx;
}
