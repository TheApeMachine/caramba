"use client";

import { useNavigate } from "@tanstack/react-router";
import { CheckIcon } from "lucide-react";
import { type TeamRow, teamCollection } from "#/collections/team";
import {
	Wizard,
	type WizardStepDefinition,
} from "#/components/ui/wizard";
import { StepDetails } from "./step-details";
import { StepMembers } from "./step-members";
import { StepPrivacy } from "./step-privacy";
import { draftFromTeam, type WizardDraft } from "./types";

const persistTeamDraft = async (teamId: string, draft: WizardDraft) => {
	const transaction = teamCollection.update(teamId, (existing) => {
		existing.description = draft.description;
		existing.color = draft.color;
		existing.emoji = draft.emoji;
		existing.privacy_mode = draft.privacyMode;
	});

	await transaction.isPersisted.promise;
};

const buildSteps = (): ReadonlyArray<WizardStepDefinition<WizardDraft>> => [
	{
		id: "details",
		title: "Team details",
		subtitle: "Give your team an identity beyond a name.",
		isComplete: () => true,
		render: ({ draft, merge }) => (
			<StepDetails draft={draft} onChange={merge} />
		),
	},
	{
		id: "members",
		title: "Invite members",
		subtitle: "Add teammates from your organization.",
		isComplete: () => true,
		render: ({ draft, merge }) => (
			<StepMembers draft={draft} onChange={merge} />
		),
	},
	{
		id: "privacy",
		title: "Privacy",
		subtitle: "Where the team's data is stored.",
		isComplete: () => true,
		render: ({ draft, merge }) => (
			<StepPrivacy draft={draft} onChange={merge} />
		),
	},
];

/*
TeamSetupWizard is the post-create setup flow. Each Continue persists
the in-progress draft through teamCollection so the user can leave
mid-flow and pick back up. Finish navigates to the team page.
*/
export const TeamSetupWizard = ({ team }: { team: TeamRow }) => {
	const navigate = useNavigate();

	return (
		<Wizard<WizardDraft>
			mode="linear"
			title={team.emoji ? `${team.emoji} ${team.name}` : team.name}
			subtitle="Setting up"
			submitLabel="Finish"
			submitPendingLabel="Finishing…"
			submitIcon={<CheckIcon />}
			steps={buildSteps()}
			initialDraft={draftFromTeam(team)}
			persistStep={(draft) => persistTeamDraft(team.id, draft)}
			onSubmit={async (draft) => {
				await persistTeamDraft(team.id, draft);
				await navigate({
					to: "/$orgSlug/$teamSlug",
					params: {
						orgSlug: team.organization_slug,
						teamSlug: team.slug,
					},
				});
			}}
		/>
	);
};
