import { useAuth } from "@clerk/tanstack-react-start";
import { useLiveQuery } from "@tanstack/react-db";
import { ClientOnly, createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRightIcon, KanbanIcon, LayersIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { researchProjectCollection } from "#/collections/research_project";
import { Badge } from "#/components/ui/badge";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

/*
KanbanHubPending shows a shared loading state for the ClientOnly
fallback and the live-query loading phase, so the swap between them
does not flash.
*/
const KanbanHubPending = () => {
	const { t } = useTranslation();

	return (
		<Flex.Center className="min-h-[60vh] p-6">
			<Typography.Paragraph variant="muted">
				{t("kanban.loadingHub")}
			</Typography.Paragraph>
		</Flex.Center>
	);
};

/*
BoardPreview is a decorative three-column kanban sketch placed inside
hub cards so each entry reads as an actual board rather than a plain
link tile.
*/
const BoardPreview = () => {
	return (
		<Flex.Row className="mt-auto h-14 gap-1.5 rounded-lg border border-border/60 bg-background/40 p-1.5">
			{[2, 3, 1].map((cardCount, columnIndex) => (
				<Flex.Column
					// biome-ignore lint/suspicious/noArrayIndexKey: static decorative columns
					key={columnIndex}
					className="h-full flex-1 gap-1 rounded-md bg-muted/40 p-1.5"
				>
					{Array.from({ length: cardCount }).map((_, cardIndex) => (
						<div
							// biome-ignore lint/suspicious/noArrayIndexKey: static decorative bars
							key={cardIndex}
							className="h-1.5 rounded-full bg-muted-foreground/25"
						/>
					))}
				</Flex.Column>
			))}
		</Flex.Row>
	);
};

/*
OrganizationHeroCard is the primary entry to the aggregate board.
It is intentionally larger than project cards so the "view everything"
path is the most discoverable.
*/
const OrganizationHeroCard = ({
	orgSlug,
	projectCount,
}: {
	orgSlug: string | null | undefined;
	projectCount: number;
}) => {
	const { t } = useTranslation();
	const resolvedSlug = orgSlug ?? "";

	return (
		<Link
			className="group relative flex flex-col gap-5 overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-primary/10 via-card to-card p-6 transition-all hover:border-primary/50 hover:shadow-lg"
			params={{ organizationSlug: resolvedSlug }}
			to="/kanban/org/$organizationSlug"
		>
			<Flex.Row className="items-center justify-between gap-4">
				<Flex.Row className="items-center gap-3">
					<div className="flex size-10 shrink-0 items-center justify-center rounded-xl border border-primary/30 bg-primary/10 text-primary">
						<LayersIcon aria-hidden className="size-5" />
					</div>
					<Flex.Column gap={1}>
						<span className="font-medium text-muted-foreground text-xs uppercase tracking-wider">
							{t("kanban.organization")}
						</span>
						<h2 className="font-semibold text-base text-foreground">
							{orgSlug ?? t("kanban.noOrganization")}
						</h2>
					</Flex.Column>
				</Flex.Row>
				<Flex.Row className="shrink-0 items-center gap-3">
					<Badge variant="outline">
						{t("kanban.projectCount", { count: projectCount })}
					</Badge>
					<ArrowRightIcon
						aria-hidden
						className="size-4 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-primary"
					/>
				</Flex.Row>
			</Flex.Row>
			<Typography.Paragraph
				className="max-w-prose text-sm"
				variant="muted"
			>
				{t("kanban.aggregateDescription")}
			</Typography.Paragraph>
			<BoardPreview />
		</Link>
	);
};

/*
ProjectBoardCard renders one project as a clickable board tile in the
grid. The org slug is rendered as monospace so the auto-generated suffix
is readable but visually demoted.
*/
const ProjectBoardCard = ({
	id,
	name,
	organizationSlug,
}: {
	id: string;
	name: string;
	organizationSlug: string | null | undefined;
}) => {
	const { t } = useTranslation();
	const slugLabel = organizationSlug?.trim()
		? organizationSlug
		: t("kanban.personalUnsorted");

	return (
		<Link
			className="group flex h-full flex-col gap-4 rounded-2xl border border-border bg-card/60 p-5 transition-all hover:-translate-y-0.5 hover:border-primary/50 hover:bg-card hover:shadow-md"
			params={{ projectId: id }}
			to="/kanban/project/$projectId"
		>
			<Flex.Row className="items-start justify-between gap-3">
				<div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-background text-muted-foreground transition-colors group-hover:border-primary/40 group-hover:text-primary">
					<KanbanIcon aria-hidden className="size-4" />
				</div>
				<ArrowRightIcon
					aria-hidden
					className="size-4 text-muted-foreground/60 transition-all group-hover:translate-x-0.5 group-hover:text-primary"
				/>
			</Flex.Row>
			<Flex.Column className="min-w-0 gap-1">
				<span className="truncate font-medium text-foreground text-sm group-hover:text-primary">
					{name}
				</span>
				<span className="truncate font-mono text-muted-foreground text-xs">
					{slugLabel}
				</span>
			</Flex.Column>
			<BoardPreview />
		</Link>
	);
};

const KanbanHubHeader = () => {
	const { t } = useTranslation();

	return (
		<Flex.Column gap={3}>
			<Flex.Row className="items-center gap-3">
				<div className="flex size-10 shrink-0 items-center justify-center rounded-xl border border-border bg-card text-primary">
					<KanbanIcon aria-hidden className="size-5" />
				</div>
				<h1 className="font-semibold text-2xl text-foreground tracking-tight">
					{t("kanban.title")}
				</h1>
			</Flex.Row>
			<Typography.Paragraph className="max-w-2xl" variant="muted">
				{t("kanban.subtitle")}
			</Typography.Paragraph>
		</Flex.Column>
	);
};

const ProjectBoardsSection = ({
	projects,
}: {
	projects: ReadonlyArray<{
		id: string;
		name: string;
		organization_slug: string | null | undefined;
	}>;
}) => {
	const { t } = useTranslation();

	return (
		<Flex.Column gap={4}>
			<Flex.Row className="items-center justify-between gap-3">
				<Flex.Row className="items-center gap-2">
					<KanbanIcon
						aria-hidden
						className="size-4 shrink-0 text-muted-foreground"
					/>
					<h2 className="font-medium text-foreground text-sm">
						{t("kanban.projectBoards")}
					</h2>
				</Flex.Row>
				<Badge variant="outline">
					{t("kanban.projectCount", { count: projects.length })}
				</Badge>
			</Flex.Row>

			{projects.length === 0 ? (
				<Flex.Center className="rounded-2xl border border-border border-dashed bg-card/30 p-12">
					<Typography.Paragraph variant="muted">
						{t("kanban.noProjects")}
					</Typography.Paragraph>
				</Flex.Center>
			) : (
				<div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
					{projects.map((project) => (
						<ProjectBoardCard
							id={project.id}
							key={project.id}
							name={project.name}
							organizationSlug={project.organization_slug}
						/>
					))}
				</div>
			)}
		</Flex.Column>
	);
};

const KanbanHubContent = () => {
	const { orgSlug } = useAuth();
	const { t } = useTranslation();

	const { data, isLoading, isError } = useLiveQuery((query) =>
		query
			.from({ project: researchProjectCollection })
			.select(({ project }) => ({
				id: project.id,
				name: project.name,
				organization_slug: project.organization_slug,
			})),
	);

	if (isLoading) {
		return <KanbanHubPending />;
	}

	if (isError) {
		return (
			<Flex.Center className="min-h-[60vh] p-6">
				<Typography.Paragraph variant="muted">
					{t("kanban.loadError")}
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	const projects = data ?? [];

	return (
		<Flex.Column className="mx-auto min-h-0 w-full max-w-6xl flex-1 gap-10 p-8">
			<KanbanHubHeader />
			<OrganizationHeroCard
				orgSlug={orgSlug}
				projectCount={projects.length}
			/>
			<ProjectBoardsSection projects={projects} />
		</Flex.Column>
	);
};

const KanbanIndexRoute = () => {
	return (
		<ClientOnly fallback={<KanbanHubPending />}>
			<KanbanHubContent />
		</ClientOnly>
	);
};

export const Route = createFileRoute("/kanban/")({
	component: KanbanIndexRoute,
});
