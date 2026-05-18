"use client";

import { FileTextIcon, PlusIcon, Trash2Icon } from "lucide-react";
import {
	createPaperDraft,
	MAX_PROJECT_PAPERS_AT_PROVISION,
	type NewResearchProjectSpec,
} from "#/components/research/new-project-model";
import { Button } from "#/components/ui/button";
import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Typography } from "#/components/ui/typography";

export const NewProjectPapersStep = ({
	spec,
	onChange,
}: {
	spec: NewResearchProjectSpec;
	onChange: (next: NewResearchProjectSpec) => void;
}) => {
	const atPaperLimit = spec.papers.length >= MAX_PROJECT_PAPERS_AT_PROVISION;

	const updatePaperTitle = (paperId: string, title: string) => {
		onChange({
			...spec,
			papers: spec.papers.map((paper) =>
				paper.id === paperId ? { ...paper, title } : paper,
			),
		});
	};

	const removePaper = (paperId: string) => {
		onChange({
			...spec,
			papers: spec.papers.filter((paper) => paper.id !== paperId),
		});
	};

	const addPaper = () => {
		if (atPaperLimit) {
			return;
		}

		const nextIndex = spec.papers.length + 1;

		onChange({
			...spec,
			papers: [...spec.papers, createPaperDraft(`Paper ${nextIndex}`)],
		});
	};

	return (
		<Flex.Column gap={3}>
			<Typography.Paragraph variant="muted">
				Link every paper this project may produce — main results, workshop
				notes, technical reports, and follow-ups. You can add more later from
				the research paper editor.
			</Typography.Paragraph>

			<ul className="flex flex-col gap-2">
				{spec.papers.map((paper, index) => (
					<li
						key={paper.id}
						className="flex flex-col gap-2 rounded-xl border bg-background/60 p-3 sm:flex-row sm:items-end"
					>
						<Field className="min-w-0 flex-1">
							<Field.Label htmlFor={`paper-title-${paper.id}`}>
								Paper {index + 1}
							</Field.Label>
							<Input
								id={`paper-title-${paper.id}`}
								value={paper.title}
								onChange={(event) =>
									updatePaperTitle(paper.id, event.target.value)
								}
								placeholder="e.g. Main conference submission"
							/>
						</Field>
						<Button
							type="button"
							variant="ghost"
							size="icon"
							aria-label={`Remove paper ${index + 1}`}
							onClick={() => removePaper(paper.id)}
						>
							<Trash2Icon className="size-4" />
						</Button>
					</li>
				))}
			</ul>

			<Flex.Row className="flex-wrap items-center gap-2">
				<Button
					type="button"
					variant="outline"
					size="sm"
					disabled={atPaperLimit}
					onClick={addPaper}
				>
					<PlusIcon className="size-4" />
					Add another paper
				</Button>
				{spec.papers.length === 0 ? (
					<Typography.Paragraph variant="muted">
						Skip papers for now — create them from the editor when you are
						ready.
					</Typography.Paragraph>
				) : null}
				{atPaperLimit ? (
					<Typography.Paragraph variant="muted">
						Up to {MAX_PROJECT_PAPERS_AT_PROVISION} papers at launch; add more
						after the project exists.
					</Typography.Paragraph>
				) : null}
			</Flex.Row>

			{spec.papers.length === 0 ? (
				<Button type="button" variant="ghost" size="sm" onClick={addPaper}>
					<FileTextIcon className="size-4" />
					Start with one paper
				</Button>
			) : null}
		</Flex.Column>
	);
};
