import { useAuth } from "@clerk/tanstack-react-start";
import { and, eq, useLiveQuery } from "@tanstack/react-db";
import { ClientOnly, createFileRoute, Link } from "@tanstack/react-router";
import {
	ActivityIcon,
	KanbanIcon,
	PlusIcon,
	SettingsIcon,
	UsersIcon,
} from "lucide-react";
import { useEffect } from "react";
import { researchProjectCollection } from "#/collections/research_project";
import { teamCollection } from "#/collections/team";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import {
	Card,
	CardFrame,
	CardFrameAction,
	CardFrameDescription,
	CardFrameHeader,
	CardFrameTitle,
	CardPanel,
} from "#/components/ui/card";
import { Empty } from "#/components/ui/empty";
import { Flex } from "#/components/ui/flex";
import { Loadable, LoadablePending } from "#/components/ui/loadable";
import { Typography } from "#/components/ui/typography";
import { setActiveTeam } from "#/lib/active-team";
import { useBreadcrumbOverride } from "#/lib/breadcrumb-overrides";
import { cn } from "#/lib/utils";

const TeamHeader = ({
	name,
	description,
	color,
	emoji,
	privacyMode,
	orgSlug,
	teamSlug,
}: {
	name: string;
	description: string;
	color: string;
	emoji: string;
	privacyMode: "shared" | "local";
	orgSlug: string;
	teamSlug: string;
}) => (
	<Flex.Column gap={4}>
		<Flex.Row align="start" justify="between" gap={4}>
			<Flex.Row align="center" gap={4}>
				<Flex.Center
					className="size-14 shrink-0 rounded-2xl border border-border bg-card text-xl"
					style={
						color ? { backgroundColor: color, color: "white" } : undefined
					}
				>
					{emoji || name.slice(0, 1).toUpperCase()}
				</Flex.Center>
				<Flex.Column gap={1}>
					<Flex.Row align="center" gap={2}>
						<Typography.PageTitle className="text-2xl">
							{name}
						</Typography.PageTitle>
						{privacyMode === "local" ? (
							<Badge size="sm" variant="warning">
								Local
							</Badge>
						) : null}
					</Flex.Row>
					{description ? (
						<Typography.Paragraph
							variant="muted"
							className="max-w-prose text-sm"
						>
							{description}
						</Typography.Paragraph>
					) : (
						<Typography.Span variant="muted" className="text-sm">
							No description yet.
						</Typography.Span>
					)}
				</Flex.Column>
			</Flex.Row>
			<Flex.Row align="center" gap={2} className="shrink-0">
				<Button render={<Link to="/research/new" />} variant="outline">
					<PlusIcon />
					New project
				</Button>
				<Button
					render={
						<Link
							params={{ orgSlug, teamSlug }}
							to="/$orgSlug/$teamSlug/setup"
						/>
					}
					variant="ghost"
				>
					<SettingsIcon />
					Settings
				</Button>
			</Flex.Row>
		</Flex.Row>
	</Flex.Column>
);

const ProjectsGrid = ({
	projects,
}: {
	projects: ReadonlyArray<{ id: string; name: string }>;
}) => (
	<CardFrame>
		<CardFrameHeader>
			<CardFrameTitle>
				<Flex.Row align="center" gap={2}>
					<KanbanIcon aria-hidden className="size-4" />
					Projects
				</Flex.Row>
			</CardFrameTitle>
			<CardFrameDescription>
				Project boards owned by this team.
			</CardFrameDescription>
			<CardFrameAction>
				<Badge size="sm" variant="outline">
					{projects.length}
				</Badge>
			</CardFrameAction>
		</CardFrameHeader>
		<Card>
			<CardPanel className="p-4">
				{projects.length === 0 ? (
					<Empty className="py-12">
						<Empty.Header>
							<Empty.Media variant="icon">
								<KanbanIcon />
							</Empty.Media>
							<Empty.Title>No projects yet</Empty.Title>
							<Empty.Description>
								Spin up your first project board to start tracking work.
							</Empty.Description>
						</Empty.Header>
						<Empty.Content>
							<Button render={<Link to="/research/new" />} variant="outline">
								<PlusIcon />
								New project
							</Button>
						</Empty.Content>
					</Empty>
				) : (
					<div
						className={cn(
							"grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3",
						)}
					>
						{projects.map((project) => (
							<Link
								className="group flex flex-col gap-2 rounded-xl border border-border bg-card p-4 transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-sm"
								key={project.id}
								params={{ projectId: project.id }}
								to="/kanban/project/$projectId"
							>
								<Flex.Row align="center" gap={2}>
									<KanbanIcon
										aria-hidden
										className="size-4 text-muted-foreground group-hover:text-primary"
									/>
									<Typography.Span truncate className="text-sm font-medium">
										{project.name}
									</Typography.Span>
								</Flex.Row>
							</Link>
						))}
					</div>
				)}
			</CardPanel>
		</Card>
	</CardFrame>
);

const ComingSoonPanel = ({
	title,
	description,
	icon: Icon,
}: {
	title: string;
	description: string;
	icon: typeof ActivityIcon;
}) => (
	<CardFrame>
		<CardFrameHeader>
			<CardFrameTitle>
				<Flex.Row align="center" gap={2}>
					<Icon aria-hidden className="size-4" />
					{title}
				</Flex.Row>
			</CardFrameTitle>
			<CardFrameDescription>{description}</CardFrameDescription>
		</CardFrameHeader>
		<Card>
			<CardPanel>
				<Empty className="py-10">
					<Empty.Header>
						<Empty.Title className="text-base">Coming soon</Empty.Title>
						<Empty.Description>
							Wiring this in is the next pass — schema and data are already in
							place.
						</Empty.Description>
					</Empty.Header>
				</Empty>
			</CardPanel>
		</Card>
	</CardFrame>
);

const TeamNotFound = ({
	orgSlug,
	teamSlug,
}: {
	orgSlug: string;
	teamSlug: string;
}) => (
	<Flex.Center fullHeight padding={6} className="min-h-[60vh]">
		<Empty>
			<Empty.Header>
				<Empty.Title>Team not found</Empty.Title>
				<Empty.Description>
					No team with slug "{teamSlug}" in "{orgSlug}".
				</Empty.Description>
			</Empty.Header>
		</Empty>
	</Flex.Center>
);

const TeamDashboardInner = () => {
	const { orgSlug, teamSlug } = Route.useParams();
	const { orgId } = useAuth();

	const teamQuery = useLiveQuery((query) =>
		query
			.from({ team: teamCollection })
			.where(({ team }) =>
				and(eq(team.organization_slug, orgSlug), eq(team.slug, teamSlug)),
			)
			.select(({ team }) => ({
				id: team.id,
				organization_slug: team.organization_slug,
				name: team.name,
				slug: team.slug,
				description: team.description,
				color: team.color,
				emoji: team.emoji,
				privacy_mode: team.privacy_mode,
			})),
	);

	const team = teamQuery.data?.[0];

	const projectsQuery = useLiveQuery(
		(query) =>
			query
				.from({ project: researchProjectCollection })
				.where(({ project }) => eq(project.team_id, team?.id ?? ""))
				.select(({ project }) => ({
					id: project.id,
					name: project.name,
				})),
		[team?.id],
	);

	useBreadcrumbOverride(`/${orgSlug}/${teamSlug}`, team?.name ?? null);

	useEffect(() => {
		if (team?.id) {
			setActiveTeam(orgId, team.id);
		}
	}, [team?.id, orgId]);

	const isLoading = teamQuery.isLoading;
	const isError = teamQuery.isError || projectsQuery.isError;
	const teamMissing = !isLoading && !isError && !team;

	return (
		<Loadable
			name="team"
			isLoading={isLoading}
			isError={isError}
			isEmpty={teamMissing}
			empty={<TeamNotFound orgSlug={orgSlug} teamSlug={teamSlug} />}
		>
			{team ? (
				<Flex.Column gap={8} padding={8} className="mx-auto w-full max-w-6xl">
					<TeamHeader
						color={team.color}
						description={team.description}
						emoji={team.emoji}
						name={team.name}
						orgSlug={team.organization_slug}
						privacyMode={team.privacy_mode}
						teamSlug={team.slug}
					/>
					<ProjectsGrid projects={projectsQuery.data ?? []} />
					<div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
						<ComingSoonPanel
							description="Latest card moves, comments, and updates across the team."
							icon={ActivityIcon}
							title="Recent activity"
						/>
						<ComingSoonPanel
							description="Every project's cards on one board, color-coded."
							icon={UsersIcon}
							title="Team kanban"
						/>
					</div>
				</Flex.Column>
			) : null}
		</Loadable>
	);
};

const TeamDashboardRoute = () => (
	<ClientOnly fallback={<LoadablePending name="team" />}>
		<TeamDashboardInner />
	</ClientOnly>
);

export const Route = createFileRoute("/$orgSlug/$teamSlug")({
	component: TeamDashboardRoute,
});
