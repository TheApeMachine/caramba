/*
Backend HTTP helpers for the Go Fiber API (port 8118 by default).

Prefer VITE_BACKEND_URL in the browser and BACKEND_URL in Nitro/server handlers.
*/
export type ClerkGetToken = () => Promise<string | null>;

/*
backendBaseURL resolves the Go API origin for client and server bundles.
*/
export function backendBaseURL(): string {
	const fromImportMeta =
		typeof import.meta !== "undefined" &&
		import.meta.env &&
		typeof import.meta.env.VITE_BACKEND_URL === "string" &&
		import.meta.env.VITE_BACKEND_URL.length > 0
			? import.meta.env.VITE_BACKEND_URL
			: "";

	if (fromImportMeta.length > 0) {
		return fromImportMeta;
	}

	const fromProcess =
		typeof process !== "undefined" &&
		process.env &&
		typeof process.env.BACKEND_URL === "string" &&
		process.env.BACKEND_URL.length > 0
			? process.env.BACKEND_URL
			: "";

	if (fromProcess.length > 0) {
		return fromProcess;
	}

	return "http://localhost:8118";
}

/*
backendAuthHeaders attaches Clerk session JWT when getToken returns a value.
*/
export async function backendAuthHeaders(
	getToken?: ClerkGetToken,
): Promise<Headers> {
	const headers = new Headers();

	if (!getToken) {
		return headers;
	}

	const token = await getToken();

	if (token) {
		headers.set("Authorization", `Bearer ${token}`);
	}

	return headers;
}
