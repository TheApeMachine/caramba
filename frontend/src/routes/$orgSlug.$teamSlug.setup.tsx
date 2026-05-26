import { and, eq, useLiveQuery } from "@tanstack/react-db";
import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { teamCollection } from "#/collections/team";
import { TeamSetupWizard } from "#/components/team-setup/wizard";
import { Empty } from "#/components/ui/empty";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const TeamSetupPending = () => {
	return (
		<Flex.Center className="min-h-[60vh] p-6">
			<Typography.Paragraph variant="muted">
				Loading team setup…
			</Typography.Paragraph>
		</Flex.Center>
	);
};

const TeamSetupInner = () => {
	const { orgSlug, teamSlug } = Route.useParams();

	const { data, isLoading } = useLiveQuery((query) =>
		query
			.from({ team: teamCollection })
			.where(({ team }) =>
				and(
					eq(team.organization_slug, orgSlug),
					eq(team.slug, teamSlug),
				),
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
				created_at: team.created_at,
				updated_at: team.updated_at,
			})),
	);

	if (isLoading) {
		return <TeamSetupPending />;
	}

	const team = data?.[0];

	if (!team) {
		return (
			<Flex.Center className="min-h-[60vh] p-6">
				<Empty>
					<Empty.Header>
						<Empty.Title>Team not found</Empty.Title>
						<Empty.Description>
							No team with slug "{teamSlug}" in "{orgSlug}". It may have been
							deleted or the URL is wrong.
						</Empty.Description>
					</Empty.Header>
				</Empty>
			</Flex.Center>
		);
	}

	return <TeamSetupWizard team={team} />;
};

const TeamSetupRoute = () => {
	return (
		<ClientOnly fallback={<TeamSetupPending />}>
			<TeamSetupInner />
		</ClientOnly>
	);
};

export const Route = createFileRoute("/$orgSlug/$teamSlug/setup")({
	component: TeamSetupRoute,
});
