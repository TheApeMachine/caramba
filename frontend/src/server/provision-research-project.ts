import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { MAX_PROJECT_PAPERS_AT_PROVISION } from "#/components/research/new-project-model";
import { backendBaseURL } from "#/lib/backend-http";

const provisionPaperSchema = z.object({
	id: z.uuid(),
	title: z.string().max(200),
});

const provisionResearchProjectInputSchema = z.object({
	id: z.uuid(),
	name: z.string().min(1),
	description: z.string(),
	project_slug: z.string().optional(),
	member_ids: z.array(z.string().min(1)),
	papers: z.array(provisionPaperSchema).max(MAX_PROJECT_PAPERS_AT_PROVISION),
});

export const provisionResearchProject = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) =>
		provisionResearchProjectInputSchema.parse(data),
	)
	.handler(async ({ data }) => {
		const authenticationState = await auth();
		const token = await authenticationState.getToken();

		if (!authenticationState.userId || !token) {
			throw new Error(
				"Research project provisioning requires a signed-in account.",
			);
		}

		const response = await fetch(
			`${backendBaseURL()}/backend/research-projects/provision`,
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
					project_slug: data.project_slug ?? "",
					member_ids: data.member_ids,
					papers: data.papers,
				}),
			},
		);

		if (!response.ok) {
			const text = await response.text();
			throw new Error(
				`Research project provision failed (${response.status}): ${text}`,
			);
		}

		const json = (await response.json()) as unknown;

		if (
			typeof json !== "object" ||
			json === null ||
			!("txid" in json) ||
			typeof (json as { txid: unknown }).txid !== "number"
		) {
			throw new Error(
				"Provision API must return JSON with a numeric txid for Electric reconciliation.",
			);
		}

		return {
			txid: (json as { txid: number }).txid,
			projectId: data.id,
			paperIds: data.papers.map((paper) => paper.id),
		};
	});
