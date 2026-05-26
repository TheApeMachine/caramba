"use client";

import { CalendarIcon, MessageSquareIcon } from "lucide-react";
import { Avatar, AvatarFallback } from "#/components/ui/avatar";
import { Badge } from "#/components/ui/badge";
import {
	Card,
	CardDescription,
	CardHeader,
	CardPanel,
	CardTitle,
} from "#/components/ui/card";
import { CardDialog } from "./card-dialog";
import { type KanbanCard, type Priority } from "./model";

interface CardItemProps {
	card: KanbanCard;
	draggable?: boolean;
	onDragStart?: (e: React.DragEvent) => void;
	onDragEnd?: (e: React.DragEvent) => void;
	isDragging?: boolean;
}

/*
PRIORITY_DOT_CLASS maps a card priority to a tailwind background color
for the small indicator dot. Kept inline to avoid coupling to Badge
variants, which carry surface-style backgrounds unsuitable for a dot.
*/
const PRIORITY_DOT_CLASS: Record<Priority, string> = {
	low: "bg-success",
	medium: "bg-info",
	high: "bg-warning",
	critical: "bg-destructive",
};

const PRIORITY_BORDER_CLASS: Record<Priority, string> = {
	low: "before:bg-success/60",
	medium: "before:bg-info/60",
	high: "before:bg-warning/70",
	critical: "before:bg-destructive/80",
};

export const CardItem = ({
	card,
	draggable,
	onDragStart,
	onDragEnd,
	isDragging,
}: CardItemProps) => {
	const overdue = card.dueDate && new Date(card.dueDate) < new Date();
	const hasMeta =
		card.description ||
		card.labels.length > 0 ||
		card.assignees.length > 0 ||
		card.dueDate;

	return (
		<CardDialog
			card={card}
			trigger={
				<Card
					className={[
						"group relative cursor-pointer select-none overflow-hidden transition-all",
						"before:pointer-events-none before:absolute before:inset-y-2 before:left-0 before:w-0.5 before:rounded-r-full",
						PRIORITY_BORDER_CLASS[card.priority],
						"hover:border-ring/40 hover:shadow-sm",
						isDragging ? "opacity-40" : "",
					].join(" ")}
					draggable={draggable}
					onDragEnd={onDragEnd}
					onDragStart={onDragStart}
				>
					<CardHeader className="gap-1.5 pb-2 pl-3.5">
						<div className="flex items-start gap-2">
							<span
								aria-label={`Priority: ${card.priority}`}
								className={[
									"mt-1.5 size-1.5 shrink-0 rounded-full",
									PRIORITY_DOT_CLASS[card.priority],
								].join(" ")}
							/>
							<CardTitle className="flex-1 text-sm leading-snug">
								{card.title}
							</CardTitle>
						</div>
						{card.sourceProjectName ? (
							<Badge className="ml-3.5 self-start" size="sm" variant="outline">
								{card.sourceProjectName}
							</Badge>
						) : null}
					</CardHeader>

					{hasMeta ? (
						<CardPanel className="flex flex-col gap-2 pb-3 pl-3.5 pt-0">
							{card.description ? (
								<CardDescription className="line-clamp-2 text-xs">
									{card.description}
								</CardDescription>
							) : null}

							{card.labels.length > 0 ? (
								<div className="flex flex-wrap gap-1">
									{card.labels.map((label) => (
										<span
											className="inline-flex items-center rounded-full px-1.5 py-0.5 font-medium text-[10px] text-white"
											key={label.id}
											style={{ backgroundColor: label.color }}
										>
											{label.text}
										</span>
									))}
								</div>
							) : null}

							{card.dueDate || card.assignees.length > 0 ? (
								<div className="flex items-center justify-between gap-2">
									{card.dueDate ? (
										<span
											className={[
												"flex items-center gap-1 text-[11px]",
												overdue
													? "text-destructive-foreground"
													: "text-muted-foreground",
											].join(" ")}
										>
											<CalendarIcon className="size-3" />
											{new Date(card.dueDate).toLocaleDateString(undefined, {
												month: "short",
												day: "numeric",
											})}
										</span>
									) : (
										<span />
									)}

									{card.assignees.length > 0 ? (
										<div className="-space-x-1.5 ml-auto flex">
											{card.assignees.slice(0, 3).map((assignee) => (
												<Avatar
													className="size-5 border border-background text-[8px]"
													key={assignee}
												>
													<AvatarFallback>
														{assignee.slice(0, 2).toUpperCase()}
													</AvatarFallback>
												</Avatar>
											))}
											{card.assignees.length > 3 ? (
												<Avatar className="size-5 border border-background text-[8px]">
													<AvatarFallback>
														+{card.assignees.length - 3}
													</AvatarFallback>
												</Avatar>
											) : null}
										</div>
									) : null}
								</div>
							) : null}
						</CardPanel>
					) : null}
				</Card>
			}
		/>
	);
};
