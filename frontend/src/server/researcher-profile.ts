import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { backendBaseURL } from "#/lib/backend-http";

export const ResearcherProfile = z.object({
	user_id: z.string(),
	display_name: z.string(),
	role_title: z.string(),
	affiliation: z.string(),
	bio: z.string(),
	website: z.string(),
	research_focus: z.string(),
	updated_at: z.string().optional(),
});

export type ResearcherProfileType = z.infer<typeof ResearcherProfile>;

const saveResearcherProfileInput = z.object({
	display_name: z.string().max(200),
	role_title: z.string().max(200),
	affiliation: z.string().max(300),
	bio: z.string().max(2000),
	website: z.string().max(500),
	research_focus: z.string().max(500),
});

export const getResearcherProfile = createServerFn({ method: "GET" }).handler(
	async () => {
		const authenticationState = await auth();
		const token = await authenticationState.getToken();

		if (!authenticationState.userId || !token) {
			throw new Error("Researcher profile requires a signed-in account.");
		}

		const response = await fetch(
			`${backendBaseURL()}/backend/researcher-profile`,
			{
				headers: { Authorization: `Bearer ${token}` },
			},
		);

		if (!response.ok) {
			const text = await response.text();
			throw new Error(
				`Researcher profile load failed (${response.status}): ${text}`,
			);
		}

		return ResearcherProfile.parse(await response.json());
	},
);

export const saveResearcherProfile = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => saveResearcherProfileInput.parse(data))
	.handler(async ({ data }) => {
		const authenticationState = await auth();
		const token = await authenticationState.getToken();

		if (!authenticationState.userId || !token) {
			throw new Error("Researcher profile requires a signed-in account.");
		}

		const response = await fetch(
			`${backendBaseURL()}/backend/researcher-profile`,
			{
				method: "PUT",
				headers: {
					Authorization: `Bearer ${token}`,
					"Content-Type": "application/json",
				},
				body: JSON.stringify(data),
			},
		);

		if (!response.ok) {
			const text = await response.text();
			throw new Error(
				`Researcher profile save failed (${response.status}): ${text}`,
			);
		}

		const json = (await response.json()) as unknown;

		if (
			typeof json !== "object" ||
			json === null ||
			!("txid" in json) ||
			typeof (json as { txid: unknown }).txid !== "number"
		) {
			throw new Error("Profile save must return a numeric txid.");
		}

		return { txid: (json as { txid: number }).txid };
	});
