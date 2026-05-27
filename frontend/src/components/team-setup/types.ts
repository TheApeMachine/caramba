import type { TeamRow } from "#/collections/team";

export type WizardDraft = {
	description: string;
	color: string;
	emoji: string;
	memberIds: ReadonlyArray<string>;
	privacyMode: "shared" | "local";
};

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
