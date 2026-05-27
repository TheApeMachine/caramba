"use client";

import { FilePlus2Icon } from "lucide-react";
import {
	type ResearchPaperRowType,
	researchPaperCollection,
} from "#/collections/research_paper";
import { Component } from "#/components/component";
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
import { useOperation } from "#/lib/use-operation";

type ProjectPaperSwitcherProps = {
	projectId: string;
	selectedPaperId?: string;
	onSelectPaperId: (paperId: string) => void;
};

const PaperPicker = ({
	papers,
	selectedPaperId,
	onSelectPaperId,
}: {
	papers: ResearchPaperRowType[];
	selectedPaperId?: string;
	onSelectPaperId: (paperId: string) => void;
}) => {
	if (papers.length === 0) {
		return (
			<Typography.Paragraph variant="muted">
				No papers linked yet. Create one to start writing.
			</Typography.Paragraph>
		);
	}

	return (
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
				{papers.map((paper) => (
					<SelectItem key={paper.id} value={paper.id}>
						{paper.title.trim() || "Untitled paper"}
					</SelectItem>
				))}
			</SelectContent>
		</Select>
	);
};

/*
ProjectPaperSwitcher lists every paper attached to the project and
exposes a single-click "New paper" action. The list is driven by a
live query on researchPaperCollection; creation status lives in a
small Tanstack Store via useOperation.
*/
export const ProjectPaperSwitcher = ({
	projectId,
	selectedPaperId,
	onSelectPaperId,
}: ProjectPaperSwitcherProps) => {
	const create = useOperation();

	const handleCreatePaper = (papers: ResearchPaperRowType[]) => {
		void create.run(async () => {
			const nextIndex = papers.length + 1;
			const paperId = await insertResearchPaperForProject(
				projectId,
				`Paper ${nextIndex}`,
			);
			onSelectPaperId(paperId);
		});
	};

	return (
		<Component<ResearchPaperRowType[]>
			name="research papers"
			isEmpty={() => false}
			query={(query) => query.from({ row: researchPaperCollection })}
		>
			{(rows) => {
				const projectPapers = rows
					.filter((row) => row.research_project_id === projectId)
					.sort(
						(left, right) =>
							left.created_at.getTime() - right.created_at.getTime(),
					);

				return (
					<Flex.Column gap={2} className="rounded-xl border bg-card/40 p-3">
						<Flex.Row align="center" wrap="wrap" gap={2}>
							<Flex.Column gap={1} className="min-w-0 flex-1">
								<Typography.H4 variant="sectionHeading">
									Research papers
								</Typography.H4>
								<Typography.Paragraph variant="muted">
									One project can host multiple distinct papers. Switch the
									active document or add another.
								</Typography.Paragraph>
							</Flex.Column>
							<Button
								type="button"
								variant="outline"
								size="sm"
								disabled={create.isPending}
								onClick={() => handleCreatePaper(projectPapers)}
							>
								<FilePlus2Icon className="size-4" />
								{create.isPending ? "Adding…" : "New paper"}
							</Button>
						</Flex.Row>

						{create.error ? (
							<Typography.Paragraph variant="error" className="text-sm">
								{create.error}
							</Typography.Paragraph>
						) : null}

						<PaperPicker
							papers={projectPapers}
							selectedPaperId={selectedPaperId}
							onSelectPaperId={onSelectPaperId}
						/>
					</Flex.Column>
				);
			}}
		</Component>
	);
};
