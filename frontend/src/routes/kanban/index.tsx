import { useAuth } from "@clerk/tanstack-react-start";
import { useLiveQuery } from "@tanstack/react-db";
import { ClientOnly, createFileRoute, Link } from "@tanstack/react-router";
import {
	ArrowRightIcon,
	KanbanIcon,
	LayersIcon,
	UsersIcon,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { researchProjectCollection } from "#/collections/research_project";
import { teamCollection } from "#/collections/team";
import { Badge } from "#/components/ui/badge";
import { Flex } from "#/components/ui/flex";
import { Loadable, LoadablePending } from "#/components/ui/loadable";
import { Typography } from "#/components/ui/typography";
import { useActiveTeam } from "#/lib/active-team";

type ProjectRow = {
	id: string;
	name: string;
	organization_slug: string | null | undefined;
	team_id: string | null | undefined;
};

type TeamRow = { id: string; name: string };

const BoardPreview = () => (
	<Flex.Row gap={2} className="mt-auto h-14 rounded-lg border border-border/60 bg-background/40 p-1.5">
		{[2, 3, 1].map((cardCount, columnIndex) => (
			<Flex.Column
				// biome-ignore lint/suspicious/noArrayIndexKey: static decorative columns
				key={columnIndex}
				gap={1}
				padding={1}
				className="h-full flex-1 rounded-md bg-muted/40"
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

const OrganizationHeroCard = ({
	orgSlug,
	projectCount,
	scopeLabel,
	scopeKind,
}: {
	orgSlug: string | null | undefined;
	projectCount: number;
	scopeLabel: string;
	scopeKind: "organization" | "team";
}) => {
	const { t } = useTranslation();
	const resolvedSlug = orgSlug ?? "";
	const ScopeIcon = scopeKind === "team" ? UsersIcon : LayersIcon;

	return (
		<Link
			className="group relative flex flex-col gap-5 overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-primary/10 via-card to-card p-6 transition-all hover:border-primary/50 hover:shadow-lg"
			params={{ organizationSlug: resolvedSlug }}
			to="/kanban/org/$organizationSlug"
		>
			<Flex.Row align="center" justify="between" gap={4}>
				<Flex.Row align="center" gap={3}>
					<Flex.Center className="size-10 shrink-0 rounded-xl border border-primary/30 bg-primary/10 text-primary">
						<ScopeIcon aria-hidden className="size-5" />
					</Flex.Center>
					<Flex.Column gap={1}>
						<Typography.Span
							variant="muted"
							className="text-xs font-medium uppercase tracking-wider"
						>
							{scopeKind === "team"
								? t("kanban.team", { defaultValue: "Team" })
								: t("kanban.organization")}
						</Typography.Span>
						<Typography.Subtitle className="text-base">
							{scopeLabel}
						</Typography.Subtitle>
					</Flex.Column>
				</Flex.Row>
				<Flex.Row align="center" gap={3} className="shrink-0">
					<Badge variant="outline">
						{t("kanban.projectCount", { count: projectCount })}
					</Badge>
					<ArrowRightIcon
						aria-hidden
						className="size-4 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-primary"
					/>
				</Flex.Row>
			</Flex.Row>
			<Typography.Paragraph variant="muted" className="max-w-prose text-sm">
				{scopeKind === "team"
					? t("kanban.teamAggregateDescription", {
							defaultValue:
								"Aggregate board across every project in this team.",
						})
					: t("kanban.aggregateDescription")}
			</Typography.Paragraph>
			<BoardPreview />
		</Link>
	);
};

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
			<Flex.Row align="start" justify="between" gap={3}>
				<Flex.Center className="size-9 shrink-0 rounded-lg border border-border bg-background text-muted-foreground transition-colors group-hover:border-primary/40 group-hover:text-primary">
					<KanbanIcon aria-hidden className="size-4" />
				</Flex.Center>
				<ArrowRightIcon
					aria-hidden
					className="size-4 text-muted-foreground/60 transition-all group-hover:translate-x-0.5 group-hover:text-primary"
				/>
			</Flex.Row>
			<Flex.Column gap={1} className="min-w-0">
				<Typography.Span
					truncate
					className="font-medium text-foreground text-sm group-hover:text-primary"
				>
					{name}
				</Typography.Span>
				<Typography.Span
					truncate
					variant="muted"
					className="font-mono text-xs"
				>
					{slugLabel}
				</Typography.Span>
			</Flex.Column>
			<BoardPreview />
		</Link>
	);
};

const KanbanHubHeader = () => {
	const { t } = useTranslation();

	return (
		<Flex.Column gap={3}>
			<Flex.Row align="center" gap={3}>
				<Flex.Center className="size-10 shrink-0 rounded-xl border border-border bg-card text-primary">
					<KanbanIcon aria-hidden className="size-5" />
				</Flex.Center>
				<Typography.PageTitle className="text-2xl tracking-tight">
					{t("kanban.title")}
				</Typography.PageTitle>
			</Flex.Row>
			<Typography.Paragraph variant="muted" className="max-w-2xl">
				{t("kanban.subtitle")}
			</Typography.Paragraph>
		</Flex.Column>
	);
};

const ProjectBoardsSection = ({
	projects,
}: {
	projects: ReadonlyArray<ProjectRow>;
}) => {
	const { t } = useTranslation();

	return (
		<Flex.Column gap={4}>
			<Flex.Row align="center" justify="between" gap={3}>
				<Flex.Row align="center" gap={2}>
					<KanbanIcon
						aria-hidden
						className="size-4 shrink-0 text-muted-foreground"
					/>
					<Typography.Subtitle className="text-sm font-medium">
						{t("kanban.projectBoards")}
					</Typography.Subtitle>
				</Flex.Row>
				<Badge variant="outline">
					{t("kanban.projectCount", { count: projects.length })}
				</Badge>
			</Flex.Row>

			{projects.length === 0 ? (
				<Flex.Center
					padding={12}
					className="rounded-2xl border border-border border-dashed bg-card/30"
				>
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
	const { orgId, orgSlug } = useAuth();
	const { t } = useTranslation();
	const activeTeamId = useActiveTeam(orgId);

	const projectsQuery = useLiveQuery((query) =>
		query.from({ project: researchProjectCollection }).select(({ project }) => ({
			id: project.id,
			name: project.name,
			organization_slug: project.organization_slug,
			team_id: project.team_id,
		})),
	);

	const teamsQuery = useLiveQuery((query) =>
		query.from({ team: teamCollection }).select(({ team }) => ({
			id: team.id,
			name: team.name,
		})),
	);

	const isLoading = projectsQuery.isLoading;
	const isError = projectsQuery.isError || teamsQuery.isError;

	const allProjects: ProjectRow[] = projectsQuery.data ?? [];
	const allTeams: TeamRow[] = teamsQuery.data ?? [];

	const projects = activeTeamId
		? allProjects.filter((project) => project.team_id === activeTeamId)
		: allProjects;

	const activeTeam = allTeams.find((team) => team.id === activeTeamId) ?? null;

	return (
		<Loadable
			name="kanban hub"
			isLoading={isLoading}
			isError={isError}
			errorMessage={t("kanban.loadError")}
		>
			<Flex.Column
				gap={10}
				padding={8}
				className="mx-auto min-h-0 w-full max-w-6xl flex-1"
			>
				<KanbanHubHeader />
				<OrganizationHeroCard
					orgSlug={orgSlug}
					projectCount={projects.length}
					scopeLabel={
						activeTeam ? activeTeam.name : (orgSlug ?? t("kanban.noOrganization"))
					}
					scopeKind={activeTeam ? "team" : "organization"}
				/>
				<ProjectBoardsSection projects={projects} />
			</Flex.Column>
		</Loadable>
	);
};

const KanbanIndexRoute = () => (
	<ClientOnly fallback={<LoadablePending name="kanban hub" />}>
		<KanbanHubContent />
	</ClientOnly>
);

export const Route = createFileRoute("/kanban/")({
	component: KanbanIndexRoute,
});
