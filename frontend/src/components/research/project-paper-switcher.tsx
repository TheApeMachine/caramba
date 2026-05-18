"use client";

import { useLiveQuery } from "@tanstack/react-db";
import { FilePlus2Icon } from "lucide-react";
import { useMemo, useState } from "react";
import type { ResearchPaperRowType } from "#/collections/research_paper";
import { researchPaperCollection } from "#/collections/research_paper";
import { insertResearchPaperForProject } from "#/components/research/insert-research-paper";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "#/components/ui/select";
import { Typography } from "#/components/ui/typography";

export const ProjectPaperSwitcher = ({
	projectId,
	selectedPaperId,
	onSelectPaperId,
}: {
	projectId: string;
	selectedPaperId?: string;
	onSelectPaperId: (paperId: string) => void;
}) => {
	const [creating, setCreating] = useState(false);
	const [createError, setCreateError] = useState<string | null>(null);

	const papersQuery = useLiveQuery((query) =>
		query.from({ row: researchPaperCollection }),
	);

	const projectPapers = useMemo(() => {
		const rows = (papersQuery.data ?? []) as ResearchPaperRowType[];

		return rows
			.filter((row) => row.research_project_id === projectId)
			.sort(
				(left, right) => left.created_at.getTime() - right.created_at.getTime(),
			);
	}, [papersQuery.data, projectId]);

	const handleCreatePaper = async () => {
		setCreateError(null);
		setCreating(true);

		try {
			const nextIndex = projectPapers.length + 1;
			const paperId = await insertResearchPaperForProject(
				projectId,
				`Paper ${nextIndex}`,
			);
			onSelectPaperId(paperId);
		} catch (error) {
			setCreateError(error instanceof Error ? error.message : String(error));
		} finally {
			setCreating(false);
		}
	};

	return (
		<Flex.Column gap={2} className="rounded-xl border bg-card/40 p-3">
			<Flex.Row className="flex-wrap items-center gap-2">
				<Flex.Column gap={1} className="min-w-0 flex-1">
					<Typography.H4 variant="sectionHeading">
						Research papers
					</Typography.H4>
					<Typography.Paragraph variant="muted">
						One project can host multiple distinct papers. Switch the active
						document or add another.
					</Typography.Paragraph>
				</Flex.Column>
				<Button
					type="button"
					variant="outline"
					size="sm"
					disabled={creating}
					onClick={() => {
						void handleCreatePaper();
					}}
				>
					<FilePlus2Icon className="size-4" />
					{creating ? "Adding…" : "New paper"}
				</Button>
			</Flex.Row>

			{createError ? (
				<Typography.Paragraph className="text-destructive text-sm">
					{createError}
				</Typography.Paragraph>
			) : null}

			{projectPapers.length === 0 ? (
				<Typography.Paragraph variant="muted">
					No papers linked yet. Create one to start writing.
				</Typography.Paragraph>
			) : (
				<Select
					value={selectedPaperId ?? ""}
					onValueChange={(value) => {
						if (value) {
							onSelectPaperId(value);
						}
					}}
				>
					<SelectTrigger className="w-full max-w-md">
						<SelectValue placeholder="Choose a paper to edit" />
					</SelectTrigger>
					<SelectContent>
						{projectPapers.map((paper) => (
							<SelectItem key={paper.id} value={paper.id}>
								{paper.title.trim() || "Untitled paper"}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
			)}
		</Flex.Column>
	);
};
