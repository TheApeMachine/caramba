"use client";

import { CalendarIcon, GripVerticalIcon } from "lucide-react";
import { Avatar, AvatarFallback } from "#/components/ui/avatar";
import { Badge } from "#/components/ui/badge";
import { Card, CardHeader, CardTitle, CardDescription, CardPanel } from "#/components/ui/card";
import { PRIORITY_COLORS, type KanbanCard } from "./model";
import { CardDialog } from "./card-dialog";

interface CardItemProps {
	card: KanbanCard;
	draggable?: boolean;
	onDragStart?: (e: React.DragEvent) => void;
	onDragEnd?: (e: React.DragEvent) => void;
	isDragging?: boolean;
}

export function CardItem({ card, draggable, onDragStart, onDragEnd, isDragging }: CardItemProps) {
	const overdue = card.dueDate && new Date(card.dueDate) < new Date();

	return (
		<CardDialog
			card={card}
			trigger={
				<Card
					draggable={draggable}
					onDragStart={onDragStart}
					onDragEnd={onDragEnd}
					className={[
						"cursor-pointer select-none transition-opacity hover:ring-1 hover:ring-ring/40",
						isDragging ? "opacity-40" : "",
					].join(" ")}
				>
					<CardHeader className="pb-2">
						<div className="flex items-start gap-2">
							<GripVerticalIcon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground/40" />
							<CardTitle className="text-sm leading-snug flex-1">{card.title}</CardTitle>
						</div>
						<Badge
							variant={PRIORITY_COLORS[card.priority]}
							size="sm"
							className="self-start ml-5"
						>
							{card.priority}
						</Badge>
					</CardHeader>

					{(card.description || card.labels.length > 0 || card.assignees.length > 0 || card.dueDate) && (
						<CardPanel className="pt-0 pb-3 flex flex-col gap-2">
							{card.description && (
								<CardDescription className="text-xs line-clamp-2 ml-5">
									{card.description}
								</CardDescription>
							)}

							{card.labels.length > 0 && (
								<div className="flex flex-wrap gap-1 ml-5">
									{card.labels.map((lbl) => (
										<span
											key={lbl.id}
											className="inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium text-white"
											style={{ backgroundColor: lbl.color }}
										>
											{lbl.text}
										</span>
									))}
								</div>
							)}

							<div className="flex items-center justify-between ml-5">
								{card.dueDate && (
									<span
										className={[
											"flex items-center gap-1 text-[11px]",
											overdue ? "text-destructive-foreground" : "text-muted-foreground",
										].join(" ")}
									>
										<CalendarIcon className="size-3" />
										{new Date(card.dueDate).toLocaleDateString()}
									</span>
								)}

								{card.assignees.length > 0 && (
									<div className="flex -space-x-1.5 ml-auto">
										{card.assignees.slice(0, 3).map((a) => (
											<Avatar key={a} className="size-5 border border-background text-[8px]">
												<AvatarFallback>{a.slice(0, 2).toUpperCase()}</AvatarFallback>
											</Avatar>
										))}
										{card.assignees.length > 3 && (
											<Avatar className="size-5 border border-background text-[8px]">
												<AvatarFallback>+{card.assignees.length - 3}</AvatarFallback>
											</Avatar>
										)}
									</div>
								)}
							</div>
						</CardPanel>
					)}
				</Card>
			}
		/>
	);
}
