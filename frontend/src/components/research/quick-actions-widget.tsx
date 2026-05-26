"use client";

import { Link } from "@tanstack/react-router";
import {
	ChevronRightIcon,
	FileTextIcon,
	FolderKanban,
	ListChecksIcon,
	PlusIcon,
	SparklesIcon,
} from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "#/components/ui/button";
import {
	Card,
	CardFrame,
	CardFrameDescription,
	CardFrameHeader,
	CardFrameTitle,
	CardPanel,
} from "#/components/ui/card";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { cn } from "@/lib/utils";

type ActionRowPosition = "first" | "middle" | "last";

const actionRowCornerClassName = (position: ActionRowPosition) =>
	cn(
		position === "first" && "rounded-b-none before:rounded-b-none",
		position === "last" && "rounded-t-none before:rounded-t-none",
		position === "middle" && "rounded-none before:rounded-none",
	);

const ActionRow = ({
	icon,
	title,
	description,
	to,
	position,
	primary = false,
}: {
	icon: ReactNode;
	title: string;
	description: string;
	to: string;
	position: ActionRowPosition;
	primary?: boolean;
}) => (
	<Button
		className={cn(
			"h-auto! w-full gap-4 px-4 py-3 text-left",
			actionRowCornerClassName(position),
		)}
		variant={primary ? "brand" : "outline"}
		render={<Link to={to} />}
	>
		{icon}
		<Flex.Column gap={1} align="start" className="min-w-0 flex-1">
			<Typography.H3>{title}</Typography.H3>
			<p
				className={cn(
					"whitespace-break-spaces font-normal",
					primary ? "text-primary-foreground" : "text-muted-foreground",
				)}
			>
				{description}
			</p>
		</Flex.Column>
		<ChevronRightIcon
			aria-hidden="true"
			className="in-[[data-slot=button]:hover]:translate-x-0.5 transition-transform"
		/>
	</Button>
);

export const QuickActionsWidget = () => (
	<CardFrame className="w-full">
		<CardFrameHeader className="flex flex-col gap-2">
			<CardFrameTitle className="flex justify-center gap-2">
				<SparklesIcon /> Quick actions
			</CardFrameTitle>
			<CardFrameDescription>
				The fastest paths into your research workspace.
			</CardFrameDescription>
		</CardFrameHeader>
		<Card>
			<CardPanel className="flex flex-col gap-0">
				<ActionRow
					position="first"
					primary
					icon={<PlusIcon className="size-5" />}
					title="New research project"
					description="Spin up a board, papers, and team in one flow."
					to="/research/new"
				/>
				<ActionRow
					position="middle"
					icon={<FileTextIcon className="size-4" />}
					title="Paper editor"
					description="Write and switch between papers linked to a project."
					to="/research/edit/research-paper"
				/>
				<ActionRow
					position="middle"
					icon={<FolderKanban className="size-4" />}
					title="Kanban boards"
					description="Track experiments, reviews, and delivery per project."
					to="/kanban"
				/>
				<ActionRow
					position="last"
					icon={<ListChecksIcon className="size-4" />}
					title="Architecture graph"
					description="Sketch and iterate on model graphs and scopes."
					to="/research/edit"
				/>
			</CardPanel>
		</Card>
	</CardFrame>
);
