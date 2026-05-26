import type { TeamRow } from "#/collections/team";

export type WizardStep = "details" | "members" | "privacy";

export type WizardDraft = {
	description: string;
	color: string;
	emoji: string;
	memberIds: ReadonlyArray<string>;
	privacyMode: "shared" | "local";
};

export const WIZARD_STEPS: ReadonlyArray<{
	id: WizardStep;
	title: string;
	subtitle: string;
}> = [
	{
		id: "details",
		title: "Team details",
		subtitle: "Give your team an identity beyond a name.",
	},
	{
		id: "members",
		title: "Invite members",
		subtitle: "Add teammates from your organization.",
	},
	{
		id: "privacy",
		title: "Privacy",
		subtitle: "Where the team's data is stored.",
	},
];

export const TEAM_COLOR_PALETTE = [
	"#3b82f6",
	"#8b5cf6",
	"#ec4899",
	"#ef4444",
	"#f97316",
	"#eab308",
	"#22c55e",
	"#14b8a6",
];

export const draftFromTeam = (team: TeamRow): WizardDraft => ({
	description: team.description ?? "",
	color: team.color ?? "",
	emoji: team.emoji ?? "",
	memberIds: [],
	privacyMode: team.privacy_mode ?? "shared",
});
