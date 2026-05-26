import { Store, useStore } from "@tanstack/react-store";

/*
ActiveTeamMap is keyed by Clerk orgId so a user's selection persists per
organization. The special key "personal" holds the selection while the
user is in their personal Clerk workspace (no active org).
*/
type ActiveTeamMap = Record<string, string | null>;

const STORAGE_KEY = "caramba.active-team";
const PERSONAL_KEY = "personal";

const readInitial = (): ActiveTeamMap => {
	if (typeof window === "undefined") {
		return {};
	}

	const raw = window.localStorage.getItem(STORAGE_KEY);

	if (!raw) {
		return {};
	}

	const parsed = JSON.parse(raw) as unknown;

	if (parsed === null || typeof parsed !== "object") {
		return {};
	}

	const result: ActiveTeamMap = {};

	for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
		if (typeof value === "string" || value === null) {
			result[key] = value;
		}
	}

	return result;
};

const activeTeamStore = new Store<ActiveTeamMap>(readInitial());

activeTeamStore.subscribe(() => {
	if (typeof window === "undefined") {
		return;
	}

	window.localStorage.setItem(
		STORAGE_KEY,
		JSON.stringify(activeTeamStore.state),
	);
});

const scopeKey = (orgId: string | null | undefined): string => {
	const trimmed = (orgId ?? "").trim();
	return trimmed === "" ? PERSONAL_KEY : trimmed;
};

/*
useActiveTeam returns the selected team id for the supplied Clerk
organization (or personal workspace when orgId is empty). Subscribes
to store updates so components re-render when the user switches.
*/
export const useActiveTeam = (
	orgId: string | null | undefined,
): string | null => {
	const key = scopeKey(orgId);
	return useStore(activeTeamStore, (state) => state[key] ?? null);
};

export const setActiveTeam = (
	orgId: string | null | undefined,
	teamId: string | null,
): void => {
	const key = scopeKey(orgId);

	activeTeamStore.setState((previous) => {
		if ((previous[key] ?? null) === teamId) {
			return previous;
		}

		return { ...previous, [key]: teamId };
	});
};
