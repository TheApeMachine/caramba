"use client";

import { useOrganization, useUser } from "@clerk/tanstack-react-start";
import { useNavigate } from "@tanstack/react-router";
import { PlayIcon } from "lucide-react";
import { useMemo } from "react";
import {
	emptyNewResearchProjectSpec,
	type NewResearchProjectSpec,
} from "#/components/research/new-project/model";
import { NewProjectPreview } from "#/components/research/new-project/preview";
import { StepBasics } from "#/components/research/new-project/step-basics";
import { StepPapers } from "#/components/research/new-project/step-papers";
import { StepReview } from "#/components/research/new-project/step-review";
import { StepTeam } from "#/components/research/new-project/step-team";
import { deriveProjectSlug } from "#/components/research/project-slug";
import { Flex } from "#/components/ui/flex";
import { Spinner } from "#/components/ui/spinner";
import { Typography } from "#/components/ui/typography";
import {
	useWizardDraft,
	Wizard,
	type WizardStepDefinition,
} from "#/components/ui/wizard";
import { provisionResearchProject } from "#/server/provision-research-project";

type PreparedContext = {
	currentUserId: string;
	currentUserLabel: string;
	memberLabels: Map<string, string>;
	initialDraft: NewResearchProjectSpec;
};

const useResolvedContext = (): PreparedContext | null => {
	const { user, isLoaded: userLoaded } = useUser();
	const { memberships, isLoaded: organizationLoaded } = useOrganization({
		memberships: { pageSize: 50 },
	});

	return useMemo(() => {
		if (!userLoaded || !organizationLoaded) {
			return null;
		}

		const currentUserId = user?.id ?? "";
		const currentUserLabel =
			[user?.firstName, user?.lastName].filter(Boolean).join(" ").trim() ||
			user?.primaryEmailAddress?.emailAddress ||
			"You";

		const memberLabels = new Map<string, string>();

		if (currentUserId) {
			memberLabels.set(currentUserId, currentUserLabel);
		}

		for (const membership of memberships?.data ?? []) {
			const id = membership.publicUserData?.userId;

			if (!id) {
				continue;
			}

			const label =
				[
					membership.publicUserData?.firstName,
					membership.publicUserData?.lastName,
				]
					.filter(Boolean)
					.join(" ")
					.trim() ||
				membership.publicUserData?.identifier ||
				id;

			memberLabels.set(id, label);
		}

		const initialDraft = emptyNewResearchProjectSpec();

		if (currentUserId) {
			initialDraft.memberIds = [currentUserId, ...initialDraft.memberIds];
		}

		return { currentUserId, currentUserLabel, memberLabels, initialDraft };
	}, [memberships?.data, organizationLoaded, user, userLoaded]);
};

const buildSteps = (
	context: PreparedContext,
): ReadonlyArray<WizardStepDefinition<NewResearchProjectSpec>> => [
	{
		id: "basics",
		title: "Project basics",
		subtitle:
			"Name and describe the research effort. The slug routes your Kanban board.",
		isComplete: (draft) => draft.name.trim().length > 0,
		render: ({ draft, merge }) => <StepBasics draft={draft} merge={merge} />,
	},
	{
		id: "team",
		title: "Team members",
		subtitle:
			"Assign collaborators. You are always included as the project owner.",
		isComplete: (draft) => draft.memberIds.length > 0,
		render: ({ draft, merge }) => (
			<StepTeam
				draft={draft}
				merge={merge}
				currentUserId={context.currentUserId}
				currentUserLabel={context.currentUserLabel}
			/>
		),
	},
	{
		id: "papers",
		title: "Research papers",
		subtitle:
			"Each paper is a distinct document linked to this project — add as many as you expect to publish.",
		isComplete: () => true,
		render: ({ draft, merge }) => <StepPapers draft={draft} merge={merge} />,
	},
	{
		id: "review",
		title: "Review",
		subtitle: "Confirm the workspace bundle before creating everything.",
		isComplete: (draft) => draft.name.trim().length > 0,
		render: ({ draft }) => <StepReview draft={draft} />,
	},
];

const LivePreview = ({
	memberLabels,
}: {
	memberLabels: Map<string, string>;
}) => {
	const draft = useWizardDraft<NewResearchProjectSpec>();
	return <NewProjectPreview spec={draft} memberLabels={memberLabels} />;
};

/*
NewProjectWizard is the sectioned wizard for spinning up a research
project. Renders a spinner until Clerk user + organization data load,
then mounts the Wizard primitive with the prepared context so the
auto-include-self logic happens once at construction (no useEffect).
*/
export const NewProjectWizard = () => {
	const navigate = useNavigate();
	const context = useResolvedContext();

	if (!context) {
		return (
			<Flex.Center fullHeight padding={8}>
				<Flex.Column align="center" gap={2}>
					<Spinner />
					<Typography.Paragraph variant="muted">
						Loading workspace…
					</Typography.Paragraph>
				</Flex.Column>
			</Flex.Center>
		);
	}

	return (
		<Wizard<NewResearchProjectSpec>
			mode="sectioned"
			title="New research project"
			subtitle="Name the effort, invite collaborators, link one or more papers, and spin up a Kanban board with starter cards in one step."
			submitLabel="Create project"
			submitPendingLabel="Creating…"
			submitIcon={<PlayIcon />}
			steps={buildSteps(context)}
			initialDraft={context.initialDraft}
			onCancel={() => navigate({ to: "/research" })}
			preview={<LivePreview memberLabels={context.memberLabels} />}
			onSubmit={async (draft) => {
				const slug = deriveProjectSlug(draft.projectSlug || draft.name);

				await provisionResearchProject({
					data: {
						id: draft.id,
						name: draft.name.trim(),
						description: draft.description.trim(),
						project_slug: slug,
						member_ids: draft.memberIds,
						papers: draft.papers.map((paper) => ({
							id: paper.id,
							title: paper.title.trim(),
						})),
					},
				});

				await navigate({
					to: "/kanban/project/$projectId",
					params: { projectId: draft.id },
				});
			}}
		/>
	);
};
