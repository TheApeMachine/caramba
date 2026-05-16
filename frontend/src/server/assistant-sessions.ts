import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { backendBaseURL } from "#/lib/backend-http";

interface SessionPayload {
	id: string;
	scope: "team" | "personal";
	title: string;
	window_size: number;
	persona_ids: string[];
}

interface MessagePayload {
	id: string;
	session_id: string;
	role: "user" | "assistant" | "system";
	parts: unknown[];
	persona_id: string;
	persona_name: string;
}

async function call(
	path: "sessions" | "messages",
	method: "POST" | "PUT" | "DELETE",
	body: unknown,
): Promise<{ txid: number }> {
	const authentication = await auth();
	const token = await authentication.getToken();

	if (!authentication.userId || !token) {
		throw new Error("Assistant writes require a signed-in account.");
	}

	const response = await fetch(`${backendBaseURL()}/backend/assistant/${path}`, {
		method,
		headers: {
			Authorization: `Bearer ${token}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify(body),
	});

	if (!response.ok) {
		throw new Error(
			`Assistant ${path} ${method} failed (${response.status}): ${await response.text()}`,
		);
	}

	const json = (await response.json()) as { txid?: unknown };

	if (typeof json.txid !== "number") {
		throw new Error(`Assistant ${path} API must return a numeric txid.`);
	}

	return { txid: json.txid };
}

export const createSession = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => data as SessionPayload)
	.handler(async ({ data }) => call("sessions", "POST", data));

export const updateSession = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => data as SessionPayload)
	.handler(async ({ data }) => call("sessions", "PUT", data));

export const deleteSession = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => data as { id: string })
	.handler(async ({ data }) => call("sessions", "DELETE", data));

export const createMessage = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => data as MessagePayload)
	.handler(async ({ data }) => call("messages", "POST", data));
