import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { ResearchProject } from "#/collections/research_project";
import { backendBaseURL } from "#/lib/backend-http";

export const createResearchProject = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => ResearchProject.parse(data))
	.handler(async ({ data }) => {
		const authenticationState = await auth();
		const token = await authenticationState.getToken();

		if (!authenticationState.userId || !token) {
			throw new Error("Research project writes require a signed-in account.");
		}

		const res = await fetch(
			`${backendBaseURL()}/backend/research-projects`,
			{
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
			},
		);

		if (!res.ok) {
			const text = await res.text();
			throw new Error(`Research project insert failed (${res.status}): ${text}`);
		}

		const json = await res.json() as unknown;

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
	});
