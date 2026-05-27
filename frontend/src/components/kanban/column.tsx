"use client";

import {
	CheckIcon,
	InboxIcon,
	MoreHorizontalIcon,
	PlusIcon,
	Trash2Icon,
} from "lucide-react";
import { useState } from "react";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import {
	Card,
	CardFrame,
	CardFrameAction,
	CardFrameDescription,
	CardFrameHeader,
	CardFrameTitle,
	CardPanel,
} from "#/components/ui/card";
import { Empty } from "#/components/ui/empty";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import {
	Menu,
	MenuItem,
	MenuPopup,
	MenuSeparator,
	MenuTrigger,
} from "#/components/ui/menu";
import { ScrollArea } from "#/components/ui/scroll-area";
import { Typography } from "#/components/ui/typography";
import { CardItem } from "./card-item";
import { useBoardContext } from "./context";
import type { KanbanColumn } from "./model";

interface KanbanColumnProps {
	column: KanbanColumn;
	onDragOver: (e: React.DragEvent, columnId: string, index: number) => void;
	onDrop: (e: React.DragEvent, columnId: string, index: number) => void;
	dragState: { cardId: string; fromColumnId: string } | null;
	columnsEditable?: boolean;
	allowAddCard?: boolean;
}

const ColumnTitleEditor = ({
	column,
	editable,
	onCommit,
}: {
	column: KanbanColumn;
	editable: boolean;
	onCommit: (next: string) => void;
}) => {
	const [editing, setEditing] = useState(false);
	const [draft, setDraft] = useState(column.title);

	if (editable && editing) {
		return (
			<Input
				autoFocus
				className="h-6 flex-1"
				onBlur={() => {
					onCommit(draft.trim());
					setEditing(false);
				}}
				onChange={(event) => setDraft(event.target.value)}
				onKeyDown={(event) => {
					if (event.key === "Enter") {
						onCommit(draft.trim());
						setEditing(false);
					}
					if (event.key === "Escape") {
						setDraft(column.title);
						setEditing(false);
					}
				}}
				size="sm"
				value={draft}
			/>
		);
	}

	if (editable) {
		return (
			<button
				className="cursor-pointer text-left text-foreground hover:text-foreground/80"
				onClick={() => setEditing(true)}
				type="button"
			>
				{column.title}
			</button>
		);
	}

	return <Typography.Span>{column.title}</Typography.Span>;
};

const ColumnMenu = ({
	atLimit,
	allowAddCard,
	onAddCard,
	onDelete,
}: {
	atLimit: boolean;
	allowAddCard: boolean;
	onAddCard: () => void;
	onDelete: () => void;
}) => {
	return (
		<Menu>
			<MenuTrigger
				aria-label="Column options"
				render={<Button size="icon-xs" variant="ghost" />}
			>
				<MoreHorizontalIcon />
			</MenuTrigger>
			<MenuPopup align="end">
				<MenuItem
					disabled={atLimit || !allowAddCard}
					onClick={() => allowAddCard && !atLimit && onAddCard()}
				>
					<PlusIcon />
					Add card
				</MenuItem>
				<MenuSeparator />
				<MenuItem
					className="text-destructive-foreground focus:bg-destructive/8"
					onClick={onDelete}
				>
					<Trash2Icon />
					Delete column
				</MenuItem>
			</MenuPopup>
		</Menu>
	);
};

const AddCardForm = ({
	onSubmit,
	onCancel,
}: {
	onSubmit: (title: string) => void;
	onCancel: () => void;
}) => {
	const [title, setTitle] = useState("");

	const commit = () => {
		const trimmed = title.trim();

		if (!trimmed) {
			onCancel();
			return;
		}

		onSubmit(trimmed);
		setTitle("");
	};

	return (
		<Flex.Column className="gap-1.5">
			<Input
				autoFocus
				onChange={(event) => setTitle(event.target.value)}
				onKeyDown={(event) => {
					if (event.key === "Enter") {
						commit();
					}
					if (event.key === "Escape") {
						setTitle("");
						onCancel();
					}
				}}
				placeholder="Card title…"
				size="sm"
				value={title}
			/>
			<Flex.Row className="gap-1">
				<Button onClick={commit} size="xs">
					<CheckIcon />
					Add
				</Button>
				<Button
					onClick={() => {
						setTitle("");
						onCancel();
					}}
					size="xs"
					variant="ghost"
				>
					Cancel
				</Button>
			</Flex.Row>
		</Flex.Column>
	);
};

export const KanbanColumnView = ({
	column,
	onDragOver,
	onDrop,
	dragState,
	columnsEditable = true,
	allowAddCard = true,
}: KanbanColumnProps) => {
	const { board, dispatch } = useBoardContext();
	const [addingCard, setAddingCard] = useState(false);
	const [dropIndex, setDropIndex] = useState<number | null>(null);

	const cards = column.cardIds.map((id) => board.cards[id]).filter(Boolean);
	const atLimit = column.limit !== null && cards.length >= column.limit;

	const handleDragOver = (e: React.DragEvent, index: number) => {
		e.preventDefault();
		setDropIndex(index);
		onDragOver(e, column.id, index);
	};

	const handleDrop = (e: React.DragEvent, index: number) => {
		setDropIndex(null);
		onDrop(e, column.id, index);
	};

	const handleAddCard = (title: string) => {
		dispatch({
			type: "ADD_CARD",
			columnId: column.id,
			card: {
				title,
				description: "",
				priority: "medium",
				labels: [],
				assignees: [],
				dueDate: null,
			},
		});
		setAddingCard(false);
	};

	return (
		<CardFrame
			className="flex h-full w-full min-w-72 flex-col"
			onDragOver={(event) => handleDragOver(event, cards.length)}
			onDrop={(event) => handleDrop(event, cards.length)}
		>
			<CardFrameHeader className="py-3">
				<CardFrameTitle>
					<Flex.Row className="items-center gap-2">
						<span
							aria-hidden
							className="size-2.5 shrink-0 rounded-full"
							style={{ backgroundColor: column.color }}
						/>
						<ColumnTitleEditor
							column={column}
							editable={columnsEditable}
							onCommit={(next) => {
								if (next) {
									dispatch({
										type: "UPDATE_COLUMN",
										id: column.id,
										title: next,
									});
								}
							}}
						/>
					</Flex.Row>
				</CardFrameTitle>
				{column.limit !== null ? (
					<CardFrameDescription className="text-xs">
						{atLimit ? "At WIP limit" : `${cards.length} of ${column.limit}`}
					</CardFrameDescription>
				) : null}
				<CardFrameAction>
					<Flex.Row className="items-center gap-1">
						<Badge size="sm" variant={atLimit ? "warning" : "outline"}>
							{cards.length}
							{column.limit !== null ? `/${column.limit}` : ""}
						</Badge>
						{columnsEditable ? (
							<ColumnMenu
								allowAddCard={allowAddCard}
								atLimit={atLimit}
								onAddCard={() => setAddingCard(true)}
								onDelete={() =>
									dispatch({ type: "DELETE_COLUMN", id: column.id })
								}
							/>
						) : null}
					</Flex.Row>
				</CardFrameAction>
			</CardFrameHeader>

			<Card className="flex min-h-0 flex-1 flex-col">
				<CardPanel className="flex min-h-0 flex-1 flex-col p-0">
					<ScrollArea className="min-h-0 flex-1">
						<Flex.Column className="gap-2 p-2">
							{cards.length === 0 && !addingCard ? (
								<Empty className="py-8">
									<Empty.Header>
										<Empty.Media variant="icon">
											<InboxIcon />
										</Empty.Media>
										<Empty.Title className="text-base">
											No cards yet
										</Empty.Title>
										<Empty.Description>
											{allowAddCard
												? "Drop a card here or add one below."
												: "Drop a card here to get started."}
										</Empty.Description>
									</Empty.Header>
									{allowAddCard ? (
										<Empty.Content>
											<Button
												disabled={atLimit}
												onClick={() => setAddingCard(true)}
												size="sm"
												variant="outline"
											>
												<PlusIcon />
												Add card
											</Button>
										</Empty.Content>
									) : null}
								</Empty>
							) : (
								cards.map((card, index) => (
									// biome-ignore lint/a11y/noStaticElementInteractions: reorder targets within a column.
									<div
										key={card.id}
										onDragOver={(event) => {
											event.stopPropagation();
											handleDragOver(event, index);
										}}
										onDrop={(event) => {
											event.stopPropagation();
											handleDrop(event, index);
										}}
									>
										{dropIndex === index &&
										dragState &&
										dragState.cardId !== card.id ? (
											<div className="mb-2 h-0.5 rounded-full bg-ring/60" />
										) : null}
										<CardItem
											card={card}
											draggable
											isDragging={dragState?.cardId === card.id}
											onDragEnd={() => {}}
											onDragStart={(event) => {
												event.dataTransfer.setData("cardId", card.id);
												event.dataTransfer.setData("fromColumnId", column.id);
											}}
										/>
									</div>
								))
							)}

							{dropIndex === cards.length && dragState ? (
								<div className="h-0.5 rounded-full bg-ring/60" />
							) : null}

							{allowAddCard && addingCard ? (
								<AddCardForm
									onCancel={() => setAddingCard(false)}
									onSubmit={handleAddCard}
								/>
							) : null}

							{cards.length > 0 && allowAddCard && !addingCard ? (
								<Button
									className="w-full justify-start text-muted-foreground"
									disabled={atLimit}
									onClick={() => setAddingCard(true)}
									size="sm"
									variant="ghost"
								>
									<PlusIcon />
									Add card
								</Button>
							) : null}
						</Flex.Column>
					</ScrollArea>
				</CardPanel>
			</Card>
		</CardFrame>
	);
};
