"use client";

import {
	CalendarIcon,
	FlagIcon,
	TagIcon,
	Trash2Icon,
	UserIcon,
	XIcon,
} from "lucide-react";
import { useState } from "react";
import { Avatar, AvatarFallback } from "#/components/ui/avatar";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import {
	Dialog,
	DialogClose,
	DialogContent,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "#/components/ui/dialog";
import { Input } from "#/components/ui/input";
import { Label } from "#/components/ui/label";
import {
	Select,
	SelectButton,
	SelectItem,
	SelectPopup,
} from "#/components/ui/select";
import { Separator } from "#/components/ui/separator";
import { Textarea } from "#/components/ui/textarea";
import { useBoardContext } from "./context";
import {
	type KanbanCard,
	LABEL_PALETTE,
	PRIORITY_COLORS,
	type Priority,
} from "./model";

const PRIORITIES: Priority[] = ["low", "medium", "high", "critical"];

interface CardDialogProps {
	card: KanbanCard;
	trigger: React.ReactNode;
}

export function CardDialog({ card, trigger }: CardDialogProps) {
	const { dispatch } = useBoardContext();
	const [title, setTitle] = useState(card.title);
	const [description, setDescription] = useState(card.description);
	const [priority, setPriority] = useState<Priority>(card.priority);
	const [dueDate, setDueDate] = useState(card.dueDate ?? "");
	const [assigneeInput, setAssigneeInput] = useState("");
	const [assignees, setAssignees] = useState<string[]>(card.assignees);
	const [labelInput, setLabelInput] = useState("");
	const [labelColor, setLabelColor] = useState(LABEL_PALETTE[0]);
	const [labels, setLabels] = useState(card.labels);

	const save = () => {
		dispatch({
			type: "UPDATE_CARD",
			id: card.id,
			changes: {
				title,
				description,
				priority,
				dueDate: dueDate || null,
				assignees,
				labels,
			},
		});
	};

	const deleteCard = () => {
		dispatch({ type: "DELETE_CARD", id: card.id });
	};

	const addAssignee = () => {
		const trimmed = assigneeInput.trim();
		if (!trimmed || assignees.includes(trimmed)) return;
		setAssignees([...assignees, trimmed]);
		setAssigneeInput("");
	};

	const addLabel = () => {
		const trimmed = labelInput.trim();
		if (!trimmed) return;
		setLabels([
			...labels,
			{ id: `${Date.now()}`, text: trimmed, color: labelColor },
		]);
		setLabelInput("");
	};

	return (
		<Dialog>
			<DialogTrigger render={<span />}>{trigger}</DialogTrigger>
			<DialogContent>
				<DialogHeader>
					<DialogTitle>Edit Card</DialogTitle>
				</DialogHeader>

				<div className="flex flex-col gap-4 p-6">
					<div className="flex flex-col gap-1.5">
						<Label>Title</Label>
						<Input value={title} onChange={(e) => setTitle(e.target.value)} />
					</div>

					<div className="flex flex-col gap-1.5">
						<Label>Description</Label>
						<Textarea
							value={description}
							onChange={(e) => setDescription(e.target.value)}
							placeholder="Add a description…"
						/>
					</div>

					<div className="grid grid-cols-2 gap-4">
						<div className="flex flex-col gap-1.5">
							<Label className="flex items-center gap-1.5">
								<FlagIcon className="size-3.5" />
								Priority
							</Label>
							<Select
								value={priority}
								onValueChange={(v) => setPriority(v as Priority)}
							>
								<SelectButton>
									<Badge variant={PRIORITY_COLORS[priority]}>{priority}</Badge>
								</SelectButton>
								<SelectPopup>
									{PRIORITIES.map((p) => (
										<SelectItem key={p} value={p}>
											<Badge variant={PRIORITY_COLORS[p]}>{p}</Badge>
										</SelectItem>
									))}
								</SelectPopup>
							</Select>
						</div>

						<div className="flex flex-col gap-1.5">
							<Label className="flex items-center gap-1.5">
								<CalendarIcon className="size-3.5" />
								Due date
							</Label>
							<Input
								nativeInput
								type="date"
								value={dueDate}
								onChange={(e) => setDueDate(e.target.value)}
							/>
						</div>
					</div>

					<Separator />

					<div className="flex flex-col gap-1.5">
						<Label className="flex items-center gap-1.5">
							<UserIcon className="size-3.5" />
							Assignees
						</Label>
						<div className="flex flex-wrap gap-1.5 mb-1">
							{assignees.map((a) => (
								<span
									key={a}
									className="flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs"
								>
									<Avatar className="size-4">
										<AvatarFallback className="text-[0.5rem]">
											{a.slice(0, 2).toUpperCase()}
										</AvatarFallback>
									</Avatar>
									{a}
									<button
										type="button"
										onClick={() =>
											setAssignees(assignees.filter((x) => x !== a))
										}
										className="opacity-60 hover:opacity-100"
									>
										<XIcon className="size-3" />
									</button>
								</span>
							))}
						</div>
						<div className="flex gap-2">
							<Input
								placeholder="Add assignee…"
								value={assigneeInput}
								onChange={(e) => setAssigneeInput(e.target.value)}
								onKeyDown={(e) => e.key === "Enter" && addAssignee()}
							/>
							<Button variant="outline" size="sm" onClick={addAssignee}>
								Add
							</Button>
						</div>
					</div>

					<div className="flex flex-col gap-1.5">
						<Label className="flex items-center gap-1.5">
							<TagIcon className="size-3.5" />
							Labels
						</Label>
						<div className="flex flex-wrap gap-1.5 mb-1">
							{labels.map((lbl) => (
								<span
									key={lbl.id}
									className="flex items-center gap-1 rounded-full px-2 py-0.5 text-xs text-white"
									style={{ backgroundColor: lbl.color }}
								>
									{lbl.text}
									<button
										type="button"
										onClick={() =>
											setLabels(labels.filter((l) => l.id !== lbl.id))
										}
										className="opacity-70 hover:opacity-100"
									>
										<XIcon className="size-3" />
									</button>
								</span>
							))}
						</div>
						<div className="flex gap-2">
							<div className="flex items-center gap-1.5 flex-1">
								<input
									type="color"
									value={labelColor}
									onChange={(e) => setLabelColor(e.target.value)}
									className="size-7 cursor-pointer rounded border border-input bg-transparent p-0.5"
								/>
								<Input
									placeholder="Label text…"
									value={labelInput}
									onChange={(e) => setLabelInput(e.target.value)}
									onKeyDown={(e) => e.key === "Enter" && addLabel()}
								/>
							</div>
							<Button variant="outline" size="sm" onClick={addLabel}>
								Add
							</Button>
						</div>
					</div>
				</div>

				<DialogFooter>
					<Button variant="destructive-outline" size="sm" onClick={deleteCard}>
						<Trash2Icon />
						Delete
					</Button>
					<DialogClose render={<Button variant="outline" />}>
						Cancel
					</DialogClose>
					<DialogClose render={<Button />} onClick={save}>
						Save
					</DialogClose>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}
