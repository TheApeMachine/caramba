"use client";

import { FileTextIcon, PlusIcon, Trash2Icon } from "lucide-react";
import {
	createPaperDraft,
	MAX_PROJECT_PAPERS_AT_PROVISION,
	type NewResearchProjectSpec,
} from "#/components/research/new-project/model";
import { Button } from "#/components/ui/button";
import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Typography } from "#/components/ui/typography";

/*
StepPapers manages an inline list of papers that will be provisioned
together with the project. The list lives entirely in the wizard
draft; nothing is written until the user submits.
*/
export const StepPapers = ({
	draft,
	merge,
}: {
	draft: NewResearchProjectSpec;
	merge: (patch: Partial<NewResearchProjectSpec>) => void;
}) => {
	const atLimit = draft.papers.length >= MAX_PROJECT_PAPERS_AT_PROVISION;

	const updatePaperTitle = (paperId: string, title: string) => {
		merge({
			papers: draft.papers.map((paper) =>
				paper.id === paperId ? { ...paper, title } : paper,
			),
		});
	};

	const removePaper = (paperId: string) => {
		merge({
			papers: draft.papers.filter((paper) => paper.id !== paperId),
		});
	};

	const addPaper = () => {
		if (atLimit) {
			return;
		}

		const nextIndex = draft.papers.length + 1;

		merge({
			papers: [...draft.papers, createPaperDraft(`Paper ${nextIndex}`)],
		});
	};

	return (
		<Flex.Column gap={3}>
			<Typography.Paragraph variant="muted">
				Link every paper this project may produce — main results, workshop
				notes, technical reports, and follow-ups. You can add more later from
				the research paper editor.
			</Typography.Paragraph>

			<Flex.Column gap={2}>
				{draft.papers.map((paper, index) => (
					<Flex.Row
						key={paper.id}
						align="end"
						gap={2}
						wrap="wrap"
						className="rounded-xl border bg-background/60 p-3"
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
					</Flex.Row>
				))}
			</Flex.Column>

			<Flex.Row align="center" wrap="wrap" gap={2}>
				<Button
					type="button"
					variant="outline"
					size="sm"
					disabled={atLimit}
					onClick={addPaper}
				>
					<PlusIcon className="size-4" />
					Add another paper
				</Button>
				{atLimit ? (
					<Typography.Paragraph variant="muted">
						Up to {MAX_PROJECT_PAPERS_AT_PROVISION} papers at launch; add more
						after the project exists.
					</Typography.Paragraph>
				) : null}
			</Flex.Row>

			{draft.papers.length === 0 ? (
				<Button type="button" variant="ghost" size="sm" onClick={addPaper}>
					<FileTextIcon className="size-4" />
					Start with one paper
				</Button>
			) : null}
		</Flex.Column>
	);
};
