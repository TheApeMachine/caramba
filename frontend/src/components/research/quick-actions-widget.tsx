"use client";

import { Link } from "@tanstack/react-router";
import {
	FileTextIcon,
	FolderKanban,
	ListChecksIcon,
	PlusIcon,
	SparklesIcon,
} from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { cn } from "@/lib/utils";

const ActionRow = ({
	icon,
	title,
	description,
	to,
	primary = false,
}: {
	icon: ReactNode;
	title: string;
	description: string;
	to: string;
	primary?: boolean;
}) => (
	<Button
		className={cn(
			"h-auto w-full justify-start gap-3 rounded-xl px-3 py-3 text-left",
			primary
				? "border-primary/30 bg-primary text-primary-foreground shadow-sm hover:bg-primary/90"
				: "border-border/80 bg-background/80 hover:bg-muted/50",
		)}
		render={<Link to={to} />}
		variant={primary ? "default" : "outline"}
	>
		<span
			className={cn(
				"flex size-10 shrink-0 items-center justify-center rounded-lg",
				primary ? "bg-primary-foreground/15" : "bg-primary/10 text-primary",
			)}
		>
			{icon}
		</span>
		<Flex.Column gap={1} className="min-w-0 flex-1 items-start">
			<span className="font-semibold text-sm">{title}</span>
			<span
				className={cn(
					"whitespace-normal font-normal text-xs leading-snug",
					primary ? "text-primary-foreground/80" : "text-muted-foreground",
				)}
			>
				{description}
			</span>
		</Flex.Column>
	</Button>
);

export const QuickActionsWidget = () => (
	<Flex.Column
		gap={3}
		className="h-full rounded-xl border border-primary/20 bg-gradient-to-b from-primary/10 via-card/40 to-card/20 p-3"
	>
		<Flex.Row className="items-center gap-2 px-1">
			<SparklesIcon className="size-4 text-primary" aria-hidden />
			<Flex.Column gap={1}>
				<Typography.H4 variant="sectionHeading">Quick actions</Typography.H4>
				<Typography.Paragraph variant="muted">
					The fastest paths into your research workspace.
				</Typography.Paragraph>
			</Flex.Column>
		</Flex.Row>

		<Flex.Column gap={2} className="min-h-0 flex-1">
			<ActionRow
				primary
				icon={<PlusIcon className="size-5" />}
				title="New research project"
				description="Spin up a board, papers, and team in one flow."
				to="/research/new"
			/>
			<ActionRow
				icon={<FileTextIcon className="size-4" />}
				title="Paper editor"
				description="Write and switch between papers linked to a project."
				to="/research/edit/research-paper"
			/>
			<ActionRow
				icon={<FolderKanban className="size-4" />}
				title="Kanban boards"
				description="Track experiments, reviews, and delivery per project."
				to="/kanban"
			/>
			<ActionRow
				icon={<ListChecksIcon className="size-4" />}
				title="Architecture graph"
				description="Sketch and iterate on model graphs and scopes."
				to="/research/edit"
			/>
		</Flex.Column>
	</Flex.Column>
);
