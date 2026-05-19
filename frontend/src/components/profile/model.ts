import type { useUser } from "@clerk/tanstack-react-start";
import type { ResearcherProfileType } from "#/server/researcher-profile";

export type ProfileExpansion = "avatar" | "identity" | "summary" | "form";

export type ProfileDraft = {
	display_name: string;
	role_title: string;
	affiliation: string;
	bio: string;
	website: string;
	research_focus: string;
};

export const profileSpring = {
	type: "spring" as const,
	stiffness: 320,
	damping: 32,
	mass: 0.7,
};

export const emptyProfileDraft = (): ProfileDraft => ({
	display_name: "",
	role_title: "",
	affiliation: "",
	bio: "",
	website: "",
	research_focus: "",
});

export const clerkProfileDefaults = (
	user: NonNullable<ReturnType<typeof useUser>["user"]>,
): ProfileDraft => ({
	display_name:
		user.fullName ??
		user.username ??
		user.primaryEmailAddress?.emailAddress ??
		"",
	role_title: "",
	affiliation: "",
	bio: "",
	website: "",
	research_focus: "",
});

export const mergeStoredProfile = (
	stored: ResearcherProfileType,
	clerkSeed: ProfileDraft,
): ProfileDraft => ({
	display_name: stored.display_name || clerkSeed.display_name,
	role_title: stored.role_title,
	affiliation: stored.affiliation,
	bio: stored.bio,
	website: stored.website,
	research_focus: stored.research_focus,
});

export const profileDisplayLabel = (draft: ProfileDraft): string =>
	draft.display_name.trim() || "Researcher";

export const profileInitials = (draft: ProfileDraft): string =>
	profileDisplayLabel(draft).slice(0, 2).toUpperCase();

export const profileDescriptionLine = (
	draft: ProfileDraft,
	fallback?: string,
): string => {
	const roleLine = [draft.role_title, draft.affiliation]
		.filter(Boolean)
		.join(" · ");

	if (roleLine) {
		return roleLine;
	}

	return fallback ?? "";
};

export const isUnsetResearcherProfile = (
	stored: ResearcherProfileType,
): boolean =>
	!stored.updated_at &&
	!stored.display_name &&
	!stored.role_title &&
	!stored.affiliation &&
	!stored.bio &&
	!stored.website &&
	!stored.research_focus;

export const nextProfileExpansion = (
	current: ProfileExpansion,
): ProfileExpansion | null => {
	if (current === "avatar") {
		return "identity";
	}

	if (current === "identity") {
		return "summary";
	}

	return null;
};
