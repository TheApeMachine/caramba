"use client";

import { PlusIcon } from "lucide-react";
import { useReducer, useState } from "react";
import { Button } from "#/components/ui/button";
import { Input } from "#/components/ui/input";
import { ScrollArea } from "#/components/ui/scroll-area";
import { BoardContext } from "./context";
import { KanbanColumnView } from "./column";
import { DEFAULT_BOARD, LABEL_PALETTE } from "./model";
import { boardReducer } from "./reducer";

export function KanbanBoard() {
	const [board, dispatch] = useReducer(boardReducer, DEFAULT_BOARD);
	const [dragState, setDragState] = useState<{ cardId: string; fromColumnId: string } | null>(null);
	const [addingColumn, setAddingColumn] = useState(false);
	const [newColumnTitle, setNewColumnTitle] = useState("");

	const handleDragStart = (e: React.DragEvent) => {
		const cardId = e.dataTransfer.getData("cardId");
		const fromColumnId = e.dataTransfer.getData("fromColumnId");
		if (cardId && fromColumnId) setDragState({ cardId, fromColumnId });
	};

	const handleDragOver = (e: React.DragEvent, _columnId: string, _index: number) => {
		e.preventDefault();
	};

	const handleDrop = (e: React.DragEvent, toColumnId: string, toIndex: number) => {
		e.preventDefault();
		const cardId = e.dataTransfer.getData("cardId");
		const fromColumnId = e.dataTransfer.getData("fromColumnId");
		if (!cardId || !fromColumnId) return;
		dispatch({ type: "MOVE_CARD", cardId, fromColumnId, toColumnId, toIndex });
		setDragState(null);
	};

	const commitNewColumn = () => {
		const trimmed = newColumnTitle.trim();
		if (!trimmed) { setAddingColumn(false); return; }
		const color = LABEL_PALETTE[board.columns.length % LABEL_PALETTE.length];
		dispatch({ type: "ADD_COLUMN", title: trimmed, color });
		setNewColumnTitle("");
		setAddingColumn(false);
	};

	return (
		<BoardContext.Provider value={{ board, dispatch }}>
			<div className="flex h-full min-h-0 flex-col">
				<ScrollArea className="flex-1">
					<ul
						className="flex min-h-0 list-none gap-3 p-4"
						style={{ minHeight: "calc(100vh - 8rem)" }}
						onDragStart={handleDragStart}
						onDragEnd={() => setDragState(null)}
					>
						{board.columns.map((col) => (
							<li key={col.id}>
								<KanbanColumnView
									column={col}
									onDragOver={handleDragOver}
									onDrop={handleDrop}
									dragState={dragState}
								/>
							</li>
						))}

						<li className="flex w-72 shrink-0 flex-col gap-1">
							{addingColumn ? (
								<div className="flex flex-col gap-2 rounded-2xl border bg-muted/40 p-3">
									<Input
										autoFocus
										size="sm"
										placeholder="Column name…"
										value={newColumnTitle}
										onChange={(e) => setNewColumnTitle(e.target.value)}
										onKeyDown={(e) => {
											if (e.key === "Enter") commitNewColumn();
											if (e.key === "Escape") { setNewColumnTitle(""); setAddingColumn(false); }
										}}
									/>
									<div className="flex gap-1">
										<Button size="xs" onClick={commitNewColumn}>Add</Button>
										<Button
											size="xs"
											variant="ghost"
											onClick={() => { setNewColumnTitle(""); setAddingColumn(false); }}
										>
											Cancel
										</Button>
									</div>
								</div>
							) : (
								<Button
									variant="outline"
									className="w-full justify-start text-muted-foreground"
									onClick={() => setAddingColumn(true)}
								>
									<PlusIcon />
									Add column
								</Button>
							)}
						</li>
					</ul>
				</ScrollArea>
			</div>
		</BoardContext.Provider>
	);
}
