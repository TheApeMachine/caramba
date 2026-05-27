"use client";

import {
	FileTextIcon,
	KanbanIcon,
	LayoutTemplateIcon,
	UsersIcon,
} from "lucide-react";
import {
	NEW_PROJECT_STARTER_CARDS,
	type NewResearchProjectSpec,
} from "#/components/research/new-project/model";
import { deriveProjectSlug } from "#/components/research/project-slug";
import { Badge } from "#/components/ui/badge";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const columnLabel: Record<string, string> = {
	backlog: "Backlog",
	todo: "To do",
	"in-progress": "In progress",
	review: "Review",
	done: "Done",
};

type PreviewProps = {
	spec: NewResearchProjectSpec;
	memberLabels: Map<string, string>;
};

const PreviewProjectCard = ({ spec }: { spec: NewResearchProjectSpec }) => {
	const slug = deriveProjectSlug(spec.projectSlug || spec.name);

	return (
		<Flex.Column gap={2} className="rounded-xl border bg-background/60 p-3">
			<Flex.Row align="center" gap={2}>
				<LayoutTemplateIcon className="size-4 text-primary" aria-hidden />
				<Typography.Span className="font-medium text-sm">
					Research project
				</Typography.Span>
			</Flex.Row>
			<Typography.Span className="font-semibold">
				{spec.name.trim() || "Untitled project"}
			</Typography.Span>
			{spec.description.trim() ? (
				<Typography.Paragraph variant="muted">
					{spec.description.trim()}
				</Typography.Paragraph>
			) : null}
			<Badge variant="outline">/{slug}</Badge>
		</Flex.Column>
	);
};

const PreviewTeamCard = ({
	memberIds,
	memberLabels,
}: {
	memberIds: ReadonlyArray<string>;
	memberLabels: Map<string, string>;
}) => {
	const rows = memberIds.length > 0 ? memberIds : ["you"];

	return (
		<Flex.Column gap={2} className="rounded-xl border bg-background/60 p-3">
			<Flex.Row align="center" gap={2}>
				<UsersIcon className="size-4 text-primary" aria-hidden />
				<Typography.Span className="font-medium text-sm">Team</Typography.Span>
				<Badge variant="secondary" className="ml-auto">
					{memberIds.length || 1} members
				</Badge>
			</Flex.Row>
			<Flex.Column gap={1}>
				{rows.map((memberId) => (
					<Typography.Span
						key={memberId}
						variant="muted"
						truncate
						className="text-sm"
					>
						{memberLabels.get(memberId) ?? memberId}
					</Typography.Span>
				))}
			</Flex.Column>
		</Flex.Column>
	);
};

const PreviewBoardCard = () => (
	<Flex.Column gap={2} className="rounded-xl border bg-background/60 p-3">
		<Flex.Row align="center" gap={2}>
			<KanbanIcon className="size-4 text-primary" aria-hidden />
			<Typography.Span className="font-medium text-sm">
				Project board
			</Typography.Span>
		</Flex.Row>
		<Flex.Column gap={2}>
			{NEW_PROJECT_STARTER_CARDS.map((card) => (
				<Flex.Column
					key={card.title}
					gap={1}
					className="rounded-lg border border-dashed bg-card/40 px-3 py-2"
				>
					<Flex.Row align="center" justify="between" gap={2}>
						<Typography.Span className="font-medium text-xs">
							{card.title}
						</Typography.Span>
						<Badge variant="outline">
							{columnLabel[card.columnKey] ?? card.columnKey}
						</Badge>
					</Flex.Row>
					<Typography.Paragraph variant="muted">
						{card.description}
					</Typography.Paragraph>
				</Flex.Column>
			))}
		</Flex.Column>
	</Flex.Column>
);

const PreviewPapersCard = ({
	papers,
}: {
	papers: NewResearchProjectSpec["papers"];
}) => (
	<Flex.Column gap={2} className="rounded-xl border bg-background/60 p-3">
		<Flex.Row align="center" gap={2}>
			<FileTextIcon className="size-4 text-primary" aria-hidden />
			<Typography.Span className="font-medium text-sm">
				Research papers
			</Typography.Span>
			<Badge variant="secondary" className="ml-auto">
				{papers.length}
			</Badge>
		</Flex.Row>
		{papers.length === 0 ? (
			<Typography.Paragraph variant="muted">
				None at launch — add papers from the editor when needed.
			</Typography.Paragraph>
		) : (
			<Flex.Column gap={1}>
				{papers.map((paper, index) => (
					<Typography.Span key={paper.id} truncate className="text-sm">
						{paper.title.trim() || `Untitled paper ${index + 1}`}
					</Typography.Span>
				))}
			</Flex.Column>
		)}
	</Flex.Column>
);

/*
NewProjectPreview is the live sidebar for the new-project wizard. Each
card mirrors a section of the draft so the user sees the workspace
bundle materialize as they fill in the form.
*/
export const NewProjectPreview = ({ spec, memberLabels }: PreviewProps) => (
	<Flex.Column
		gap={4}
		fullHeight
		className="min-h-0 rounded-2xl border bg-card/50 p-4"
	>
		<Flex.Column gap={1}>
			<Typography.H3 variant="sectionHeading">Workspace preview</Typography.H3>
			<Typography.Paragraph variant="muted">
				Everything below is created in one transaction when you launch.
			</Typography.Paragraph>
		</Flex.Column>

		<PreviewProjectCard spec={spec} />
		<PreviewTeamCard memberIds={spec.memberIds} memberLabels={memberLabels} />
		<PreviewBoardCard />
		<PreviewPapersCard papers={spec.papers} />
	</Flex.Column>
);
