function stripTrailingSlash(s: string): string {
	return s.endsWith("/") ? s.slice(0, -1) : s;
}

function stripLeadingSlash(s: string): string {
	return s.startsWith("/") ? s.slice(1) : s;
}

/**
 * Build an API URL for the control-plane backend.
 *
 * - In dev, we default to http://127.0.0.1:8765 to avoid relying on a dev proxy.
 * - Override with VITE_API_BASE (e.g. http://localhost:8765).
 * - In prod, default is "" (same-origin).
 */
export function apiBase(): string {
	const envBase = (import.meta as ImportMeta & { env?: Record<string, unknown> })
		.env?.VITE_API_BASE;
	if (typeof envBase === "string" && envBase.length > 0) {
		return stripTrailingSlash(envBase);
	}
	// Default for local dev: bypass the Vite/TanStack server proxy stack.
	if ((import.meta as ImportMeta & { env?: Record<string, unknown> }).env?.DEV) {
		return "http://127.0.0.1:8765";
	}
	return "";
}

export function apiUrl(path: string): string {
	const base = apiBase();
	if (!base) return path;
	return `${base}/${stripLeadingSlash(path)}`;
}

