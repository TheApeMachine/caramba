import { useCallback, useEffect, useState } from "react";

const MODE_KEY = "caramba:assistant:local-only";
const LOCAL_ENDPOINT_KEY = "caramba:assistant:local-endpoint";
const LOCAL_AUTH_KEY = "caramba:assistant:local-auth";

export type AssistantBackendMode = "cloud" | "local";

export interface LocalEndpointConfig {
	baseURL: string;
	authHeader: string;
}

function readMode(): AssistantBackendMode {
	if (typeof window === "undefined") return "cloud";
	return window.localStorage.getItem(MODE_KEY) === "local" ? "local" : "cloud";
}

function readEndpoint(): LocalEndpointConfig {
	if (typeof window === "undefined") return { baseURL: "", authHeader: "" };
	return {
		baseURL: window.localStorage.getItem(LOCAL_ENDPOINT_KEY) ?? "",
		authHeader: window.localStorage.getItem(LOCAL_AUTH_KEY) ?? "",
	};
}

/*
useAssistantMode persists the local-only flag and the local endpoint config
(used by per-persona ollama/openai-compat adapters). Persisted in
localStorage so the choice survives reloads without touching the database.
*/
export function useAssistantMode() {
	const [mode, setMode] = useState<AssistantBackendMode>(() => readMode());
	const [endpoint, setEndpoint] = useState<LocalEndpointConfig>(() => readEndpoint());

	useEffect(() => {
		if (typeof window === "undefined") return;
		window.localStorage.setItem(MODE_KEY, mode);
	}, [mode]);

	useEffect(() => {
		if (typeof window === "undefined") return;
		window.localStorage.setItem(LOCAL_ENDPOINT_KEY, endpoint.baseURL);
		window.localStorage.setItem(LOCAL_AUTH_KEY, endpoint.authHeader);
	}, [endpoint]);

	const toggle = useCallback(() => {
		setMode((current) => (current === "cloud" ? "local" : "cloud"));
	}, []);

	const updateEndpoint = useCallback((next: Partial<LocalEndpointConfig>) => {
		setEndpoint((current) => ({ ...current, ...next }));
	}, []);

	return { mode, setMode, toggle, endpoint, updateEndpoint };
}
