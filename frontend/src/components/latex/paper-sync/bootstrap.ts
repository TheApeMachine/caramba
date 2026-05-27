const DRAFT_STORAGE_PREFIX = "caramba:research-paper-bootstrap:";

const UUID_RE =
	/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

const storageKey = (projectId: string): string =>
	`${DRAFT_STORAGE_PREFIX}${projectId}`;

/*
readBootstrapDraftId returns the paper id this client previously
reserved for the project (if any). Used so multiple route enters
during the same browser session pick up the same in-flight bootstrap
instead of creating duplicate untitled papers.
*/
export const readBootstrapDraftId = (projectId: string): string | null => {
	if (typeof window === "undefined") {
		return null;
	}

	const raw = window.sessionStorage.getItem(storageKey(projectId));

	if (!raw || !UUID_RE.test(raw)) {
		return null;
	}

	return raw;
};

export const writeBootstrapDraftId = (
	projectId: string,
	paperId: string,
): void => {
	if (typeof window === "undefined") {
		return;
	}

	window.sessionStorage.setItem(storageKey(projectId), paperId);
};

export const clearBootstrapDraftId = (projectId: string): void => {
	if (typeof window === "undefined") {
		return;
	}

	window.sessionStorage.removeItem(storageKey(projectId));
};
