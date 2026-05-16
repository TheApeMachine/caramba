import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { backendBaseURL } from "#/lib/backend-http";

interface PersonaPayload {
	id: string;
	scope: "global" | "team" | "personal";
	name: string;
	system_prompt: string;
	model: string;
	temperature: number;
	max_tokens: number;
	adapter_type: "openai" | "ollama" | "openai-compat";
	endpoint_url: string;
}

interface DeletePayload {
	id: string;
	scope: PersonaPayload["scope"];
}

async function call(
	method: "POST" | "PUT" | "DELETE",
	body: unknown,
): Promise<{ txid: number }> {
	const authentication = await auth();
	const token = await authentication.getToken();

	if (!authentication.userId || !token) {
		throw new Error("Persona writes require a signed-in account.");
	}

	const response = await fetch(`${backendBaseURL()}/backend/assistant/personas`, {
		method,
		headers: {
			Authorization: `Bearer ${token}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify(body),
	});

	if (!response.ok) {
		throw new Error(
			`Persona ${method} failed (${response.status}): ${await response.text()}`,
		);
	}

	const json = (await response.json()) as { txid?: unknown };

	if (typeof json.txid !== "number") {
		throw new Error("Persona API must return a numeric txid.");
	}

	return { txid: json.txid };
}

export const createPersona = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => data as PersonaPayload)
	.handler(async ({ data }) => call("POST", data));

export const updatePersona = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => data as PersonaPayload)
	.handler(async ({ data }) => call("PUT", data));

export const deletePersona = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => data as DeletePayload)
	.handler(async ({ data }) => call("DELETE", data));
