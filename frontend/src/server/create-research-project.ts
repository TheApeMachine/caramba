import { createServerFn } from "@tanstack/react-start";
import { auth } from "@clerk/tanstack-react-start/server";
import { backendBaseURL } from "#/lib/backend-http";
import { researchProjectRowSchema } from "#/lib/research-project-schema";

async function insertResearchProjectViaHttp(
	url: string,
	data: ReturnType<typeof researchProjectRowSchema.parse>,
	token: string,
): Promise<{ txid: number }> {
	const res = await fetch(url, {
		method: "POST",
		headers: {
			Authorization: `Bearer ${token}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			id: data.id,
			name: data.name,
			description: data.description,
			project_slug: data.project_slug ?? null,
		}),
	});

	if (!res.ok) {
		const text = await res.text();
		throw new Error(`Research project insert failed (${res.status}): ${text}`);
	}

	const json: unknown = await res.json();
	if (
		typeof json !== "object" ||
		json === null ||
		!("txid" in json) ||
		typeof (json as { txid: unknown }).txid !== "number"
	) {
		throw new Error(
			"Insert API must return JSON with a numeric txid for Electric reconciliation.",
		);
	}
	return json as { txid: number };
}

/**
 * Persists a research project for Electric reconciliation:
 * POSTs to the Go backend so organization and admin authorization are derived
 * from the Clerk session instead of client-submitted fields.
 */
export const createResearchProject = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => researchProjectRowSchema.parse(data))
	.handler(async ({ data }) => {
		const authenticationState = await auth();
		const token = await authenticationState.getToken();

		if (!authenticationState.userId || !token) {
			throw new Error("Research project writes require a signed-in account.");
		}

		return insertResearchProjectViaHttp(
			`${backendBaseURL()}/backend/research-projects`,
			data,
			token,
		);
	});
