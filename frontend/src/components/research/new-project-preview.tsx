import {
	FileTextIcon,
	KanbanIcon,
	LayoutTemplateIcon,
	UsersIcon,
} from "lucide-react";
import {
	NEW_PROJECT_STARTER_CARDS,
	type NewResearchProjectSpec,
} from "#/components/research/new-project-model";
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

export const NewProjectPreview = ({
	spec,
	memberLabels,
}: {
	spec: NewResearchProjectSpec;
	memberLabels: Map<string, string>;
}) => {
	const slug = deriveProjectSlug(spec.projectSlug || spec.name);

	return (
		<Flex.Column
			gap={4}
			className="h-full min-h-0 rounded-2xl border bg-card/50 p-4"
		>
			<Flex.Column gap={1}>
				<Typography.H3 variant="sectionHeading">
					Workspace preview
				</Typography.H3>
				<Typography.Paragraph variant="muted">
					Everything below is created in one transaction when you launch.
				</Typography.Paragraph>
			</Flex.Column>

			<Flex.Column gap={2} className="rounded-xl border bg-background/60 p-3">
				<Flex.Row className="items-center gap-2">
					<LayoutTemplateIcon className="size-4 text-primary" aria-hidden />
					<span className="font-medium text-sm">Research project</span>
				</Flex.Row>
				<span className="font-semibold text-foreground">
					{spec.name.trim() || "Untitled project"}
				</span>
				{spec.description.trim() ? (
					<Typography.Paragraph variant="muted">
						{spec.description.trim()}
					</Typography.Paragraph>
				) : null}
				<Badge variant="outline">/{slug}</Badge>
			</Flex.Column>

			<Flex.Column gap={2} className="rounded-xl border bg-background/60 p-3">
				<Flex.Row className="items-center gap-2">
					<UsersIcon className="size-4 text-primary" aria-hidden />
					<span className="font-medium text-sm">Team</span>
					<Badge variant="secondary" className="ml-auto">
						{spec.memberIds.length || 1} members
					</Badge>
				</Flex.Row>
				<ul className="flex flex-col gap-1">
					{(spec.memberIds.length > 0 ? spec.memberIds : ["you"]).map(
						(memberId) => (
							<li
								key={memberId}
								className="truncate text-muted-foreground text-sm"
							>
								{memberLabels.get(memberId) ?? memberId}
							</li>
						),
					)}
				</ul>
			</Flex.Column>

			<Flex.Column gap={2} className="rounded-xl border bg-background/60 p-3">
				<Flex.Row className="items-center gap-2">
					<KanbanIcon className="size-4 text-primary" aria-hidden />
					<span className="font-medium text-sm">Project board</span>
				</Flex.Row>
				<ul className="flex flex-col gap-2">
					{NEW_PROJECT_STARTER_CARDS.map((card) => (
						<li
							key={card.title}
							className="rounded-lg border border-dashed bg-card/40 px-3 py-2"
						>
							<Flex.Row className="items-center justify-between gap-2">
								<span className="font-medium text-foreground text-xs">
									{card.title}
								</span>
								<Badge variant="outline">
									{columnLabel[card.columnKey] ?? card.columnKey}
								</Badge>
							</Flex.Row>
							<p className="mt-1 text-muted-foreground text-xs leading-snug">
								{card.description}
							</p>
						</li>
					))}
				</ul>
			</Flex.Column>

			<Flex.Column gap={2} className="rounded-xl border bg-background/60 p-3">
				<Flex.Row className="items-center gap-2">
					<FileTextIcon className="size-4 text-primary" aria-hidden />
					<span className="font-medium text-sm">Research papers</span>
					<Badge variant="secondary" className="ml-auto">
						{spec.papers.length}
					</Badge>
				</Flex.Row>
				{spec.papers.length === 0 ? (
					<Typography.Paragraph variant="muted">
						None at launch — add papers from the editor when needed.
					</Typography.Paragraph>
				) : (
					<ul className="flex flex-col gap-1">
						{spec.papers.map((paper, index) => (
							<li key={paper.id} className="truncate text-foreground text-sm">
								{paper.title.trim() || `Untitled paper ${index + 1}`}
							</li>
						))}
					</ul>
				)}
			</Flex.Column>
		</Flex.Column>
	);
};
