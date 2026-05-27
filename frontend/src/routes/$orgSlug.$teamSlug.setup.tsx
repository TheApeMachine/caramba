import { and, eq } from "@tanstack/react-db";
import { createFileRoute } from "@tanstack/react-router";
import { teamCollection } from "#/collections/team";
import { Component } from "#/components/component";
import { TeamSetupWizard } from "#/components/team-setup/wizard";
import { Empty } from "#/components/ui/empty";
import { Flex } from "#/components/ui/flex";

type TeamSetupRow = {
	id: string;
	organization_slug: string;
	name: string;
	slug: string;
	description: string;
	color: string;
	emoji: string;
	privacy_mode: "shared" | "local";
	created_at: Date;
	updated_at: Date;
};

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
					No team with slug "{teamSlug}" in "{orgSlug}". It may have been
					deleted or the URL is wrong.
				</Empty.Description>
			</Empty.Header>
		</Empty>
	</Flex.Center>
);

const TeamSetupRoute = () => {
	const { orgSlug, teamSlug } = Route.useParams();

	return (
		<Component<TeamSetupRow[]>
			name="team setup"
			isEmpty={(rows) => rows.length === 0}
			empty={<TeamNotFound orgSlug={orgSlug} teamSlug={teamSlug} />}
			query={(query) =>
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
						created_at: team.created_at,
						updated_at: team.updated_at,
					}))
			}
		>
			{(rows) => <TeamSetupWizard team={rows[0]} />}
		</Component>
	);
};

export const Route = createFileRoute("/$orgSlug/$teamSlug/setup")({
	component: TeamSetupRoute,
});
