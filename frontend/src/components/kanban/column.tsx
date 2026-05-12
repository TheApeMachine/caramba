"use client";

import { CheckIcon, MoreHorizontalIcon, PlusIcon, Trash2Icon } from "lucide-react";
import { useState } from "react";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import { Input } from "#/components/ui/input";
import { Menu, MenuItem, MenuPopup, MenuSeparator, MenuTrigger } from "#/components/ui/menu";
import { ScrollArea } from "#/components/ui/scroll-area";
import { useBoardContext } from "./context";
import type { KanbanColumn } from "./model";
import { CardItem } from "./card-item";

interface KanbanColumnProps {
	column: KanbanColumn;
	onDragOver: (e: React.DragEvent, columnId: string, index: number) => void;
	onDrop: (e: React.DragEvent, columnId: string, index: number) => void;
	dragState: { cardId: string; fromColumnId: string } | null;
}

export function KanbanColumnView({ column, onDragOver, onDrop, dragState }: KanbanColumnProps) {
	const { board, dispatch } = useBoardContext();
	const [addingCard, setAddingCard] = useState(false);
	const [newCardTitle, setNewCardTitle] = useState("");
	const [editingTitle, setEditingTitle] = useState(false);
	const [titleInput, setTitleInput] = useState(column.title);
	const [dropIndex, setDropIndex] = useState<number | null>(null);

	const cards = column.cardIds
		.map((id) => board.cards[id])
		.filter(Boolean);

	const atLimit = column.limit !== null && cards.length >= column.limit;

	const commitNewCard = () => {
		const trimmed = newCardTitle.trim();
		if (!trimmed) { setAddingCard(false); return; }
		dispatch({
			type: "ADD_CARD",
			columnId: column.id,
			card: { title: trimmed, description: "", priority: "medium", labels: [], assignees: [], dueDate: null },
		});
		setNewCardTitle("");
		setAddingCard(false);
	};

	const commitTitle = () => {
		const trimmed = titleInput.trim();
		if (trimmed) dispatch({ type: "UPDATE_COLUMN", id: column.id, title: trimmed });
		setEditingTitle(false);
	};

	const handleDragOver = (e: React.DragEvent, index: number) => {
		e.preventDefault();
		setDropIndex(index);
		onDragOver(e, column.id, index);
	};

	const handleDrop = (e: React.DragEvent, index: number) => {
		setDropIndex(null);
		onDrop(e, column.id, index);
	};

	return (
		<div
			className="flex w-72 shrink-0 flex-col rounded-2xl border bg-muted/40"
			onDragOver={(e) => handleDragOver(e, cards.length)}
			onDrop={(e) => handleDrop(e, cards.length)}
		>
			<div className="flex items-center gap-2 px-3 py-2.5">
				<span
					className="size-2.5 rounded-full shrink-0"
					style={{ backgroundColor: column.color }}
				/>

				{editingTitle ? (
					<Input
						autoFocus
						size="sm"
						value={titleInput}
						onChange={(e) => setTitleInput(e.target.value)}
						onBlur={commitTitle}
						onKeyDown={(e) => {
							if (e.key === "Enter") commitTitle();
							if (e.key === "Escape") { setTitleInput(column.title); setEditingTitle(false); }
						}}
						className="h-6 flex-1"
					/>
				) : (
					<button
						type="button"
						className="flex-1 text-left text-sm font-semibold hover:text-foreground/80"
						onClick={() => setEditingTitle(true)}
					>
						{column.title}
					</button>
				)}

				<div className="flex items-center gap-1">
					<Badge variant="outline" size="sm">
						{cards.length}{column.limit !== null ? `/${column.limit}` : ""}
					</Badge>

					<Menu>
						<MenuTrigger
							render={<Button size="icon-xs" variant="ghost" />}
							aria-label="Column options"
						>
							<MoreHorizontalIcon />
						</MenuTrigger>
						<MenuPopup align="end">
							<MenuItem
								disabled={atLimit}
								onClick={() => !atLimit && setAddingCard(true)}
							>
								<PlusIcon />
								Add card
							</MenuItem>
							<MenuSeparator />
							<MenuItem
								className="text-destructive-foreground focus:bg-destructive/8"
								onClick={() => dispatch({ type: "DELETE_COLUMN", id: column.id })}
							>
								<Trash2Icon />
								Delete column
							</MenuItem>
						</MenuPopup>
					</Menu>
				</div>
			</div>

			<ScrollArea className="flex-1 px-2">
				<div className="flex flex-col gap-2 py-1 pb-2">
					{cards.map((card, index) => (
						<div
							key={card.id}
							onDragOver={(e) => { e.stopPropagation(); handleDragOver(e, index); }}
							onDrop={(e) => { e.stopPropagation(); handleDrop(e, index); }}
						>
							{dropIndex === index && dragState && dragState.cardId !== card.id && (
								<div className="mb-2 h-0.5 rounded-full bg-ring/60" />
							)}
							<CardItem
								card={card}
								draggable
								isDragging={dragState?.cardId === card.id}
								onDragStart={(e) => {
									e.dataTransfer.setData("cardId", card.id);
									e.dataTransfer.setData("fromColumnId", column.id);
								}}
								onDragEnd={() => {}}
							/>
						</div>
					))}

					{dropIndex === cards.length && dragState && (
						<div className="h-0.5 rounded-full bg-ring/60" />
					)}

					{addingCard ? (
						<div className="flex flex-col gap-1.5">
							<Input
								autoFocus
								size="sm"
								placeholder="Card title…"
								value={newCardTitle}
								onChange={(e) => setNewCardTitle(e.target.value)}
								onKeyDown={(e) => {
									if (e.key === "Enter") commitNewCard();
									if (e.key === "Escape") { setNewCardTitle(""); setAddingCard(false); }
								}}
							/>
							<div className="flex gap-1">
								<Button size="xs" onClick={commitNewCard}>
									<CheckIcon />
									Add
								</Button>
								<Button
									size="xs"
									variant="ghost"
									onClick={() => { setNewCardTitle(""); setAddingCard(false); }}
								>
									Cancel
								</Button>
							</div>
						</div>
					) : (
						<Button
							variant="ghost"
							size="sm"
							className="w-full justify-start text-muted-foreground"
							disabled={atLimit}
							onClick={() => setAddingCard(true)}
						>
							<PlusIcon />
							Add card
						</Button>
					)}
				</div>
			</ScrollArea>
		</div>
	);
}
