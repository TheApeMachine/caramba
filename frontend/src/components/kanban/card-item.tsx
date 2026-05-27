"use client";

import { CalendarIcon, ClockIcon } from "lucide-react";
import { Avatar, AvatarFallback } from "#/components/ui/avatar";
import { Badge } from "#/components/ui/badge";
import {
	Card,
	CardDescription,
	CardHeader,
	CardPanel,
	CardTitle,
} from "#/components/ui/card";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { relativeTime } from "#/lib/relative-time";
import { cn } from "#/lib/utils";
import { CardDialog } from "./card-dialog";
import type { KanbanCard, Priority } from "./model";

interface CardItemProps {
	card: KanbanCard;
	draggable?: boolean;
	onDragStart?: (e: React.DragEvent) => void;
	onDragEnd?: (e: React.DragEvent) => void;
	isDragging?: boolean;
}

/*
PRIORITY_CHIP maps each priority to the chip dot color and pill
background. Backgrounds are tinted at low opacity so the chip reads
as metadata without competing with the title for attention.
*/
const PRIORITY_CHIP: Record<
	Priority,
	{ dot: string; pill: string; label: string }
> = {
	low: {
		dot: "bg-success",
		pill: "bg-success/10 text-success-foreground",
		label: "Low",
	},
	medium: {
		dot: "bg-info",
		pill: "bg-info/10 text-info-foreground",
		label: "Medium",
	},
	high: {
		dot: "bg-warning",
		pill: "bg-warning/10 text-warning-foreground",
		label: "High",
	},
	critical: {
		dot: "bg-destructive",
		pill: "bg-destructive/12 text-destructive-foreground",
		label: "Critical",
	},
};

const PriorityChip = ({ priority }: { priority: Priority }) => {
	const chip = PRIORITY_CHIP[priority];

	return (
		<Flex.Row
			aria-label={`Priority: ${chip.label}`}
			className={cn(
				"shrink-0 items-center gap-1.5 rounded-full px-2 py-0.5",
				chip.pill,
			)}
		>
			<span className={cn("size-1.5 rounded-full", chip.dot)} />
			<Typography.Span className="text-[10px] font-medium uppercase tracking-wide">
				{chip.label}
			</Typography.Span>
		</Flex.Row>
	);
};

const AssigneeStack = ({ assignees }: { assignees: ReadonlyArray<string> }) => {
	const shown = assignees.slice(0, 3);
	const overflow = assignees.length - shown.length;

	return (
		<Flex.Row className="-space-x-1.5 ml-auto">
			{shown.map((assignee) => (
				<Avatar
					className="size-5 border border-background text-[8px]"
					key={assignee}
				>
					<AvatarFallback>{assignee.slice(0, 2).toUpperCase()}</AvatarFallback>
				</Avatar>
			))}
			{overflow > 0 ? (
				<Avatar className="size-5 border border-background text-[8px]">
					<AvatarFallback>+{overflow}</AvatarFallback>
				</Avatar>
			) : null}
		</Flex.Row>
	);
};

export const CardItem = ({
	card,
	draggable,
	onDragStart,
	onDragEnd,
	isDragging,
}: CardItemProps) => {
	const overdue = card.dueDate && new Date(card.dueDate) < new Date();

	return (
		<CardDialog
			card={card}
			trigger={
				<Card
					className={cn(
						"group cursor-pointer select-none border-border bg-card shadow-sm/10 transition-all",
						"hover:border-ring/50 hover:shadow-md",
						isDragging && "opacity-40",
					)}
					draggable={draggable}
					onDragEnd={onDragEnd}
					onDragStart={onDragStart}
				>
					<CardHeader className="gap-2.5 pb-2">
						<Flex.Row className="items-start justify-between gap-2">
							<CardTitle className="flex-1 text-sm leading-snug text-foreground">
								{card.title}
							</CardTitle>
							<PriorityChip priority={card.priority} />
						</Flex.Row>
						{card.sourceProjectName ? (
							<Badge className="self-start" size="sm" variant="outline">
								{card.sourceProjectName}
							</Badge>
						) : null}
					</CardHeader>

					<CardPanel className="pt-0 pb-3">
						<Flex.Column className="gap-2">
							{card.description ? (
								<CardDescription className="line-clamp-2 text-xs">
									{card.description}
								</CardDescription>
							) : null}

							{card.labels.length > 0 ? (
								<Flex.Row className="flex-wrap gap-1">
									{card.labels.map((label) => (
										<span
											className="inline-flex items-center rounded-full px-1.5 py-0.5 font-medium text-[10px] text-white"
											key={label.id}
											style={{ backgroundColor: label.color }}
										>
											{label.text}
										</span>
									))}
								</Flex.Row>
							) : null}

							<Flex.Row className="items-center justify-between gap-2">
								<Flex.Row className="items-center gap-3 text-muted-foreground">
									<Flex.Row className="items-center gap-1 text-[11px]">
										<ClockIcon className="size-3" />
										<Typography.Span variant="muted">
											{relativeTime(card.createdAt)}
										</Typography.Span>
									</Flex.Row>
									{card.dueDate ? (
										<Flex.Row
											className={cn(
												"items-center gap-1 text-[11px]",
												overdue
													? "text-destructive-foreground"
													: "text-muted-foreground",
											)}
										>
											<CalendarIcon className="size-3" />
											<Typography.Span variant={overdue ? "error" : "muted"}>
												{new Date(card.dueDate).toLocaleDateString(undefined, {
													month: "short",
													day: "numeric",
												})}
											</Typography.Span>
										</Flex.Row>
									) : null}
								</Flex.Row>
								{card.assignees.length > 0 ? (
									<AssigneeStack assignees={card.assignees} />
								) : null}
							</Flex.Row>
						</Flex.Column>
					</CardPanel>
				</Card>
			}
		/>
	);
};
