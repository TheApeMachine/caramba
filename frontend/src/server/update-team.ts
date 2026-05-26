import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { backendBaseURL } from "#/lib/backend-http";

const UpdateTeamPayload = z.object({
	id: z.uuid(),
	name: z.string().min(1).optional(),
	description: z.string().optional(),
	color: z.string().optional(),
	emoji: z.string().optional(),
	privacy_mode: z.enum(["shared", "local"]).optional(),
});

export const updateTeam = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => UpdateTeamPayload.parse(data))
	.handler(async ({ data }) => {
		const authenticationState = await auth();
		const token = await authenticationState.getToken();

		if (!authenticationState.userId || !token) {
			throw new Error("Team writes require a signed-in account.");
		}

		const response = await fetch(`${backendBaseURL()}/backend/teams`, {
			method: "PUT",
			headers: {
				Authorization: `Bearer ${token}`,
				"Content-Type": "application/json",
			},
			body: JSON.stringify(data),
		});

		if (!response.ok) {
			const text = await response.text();
			throw new Error(`Team update failed (${response.status}): ${text}`);
		}

		const json = (await response.json()) as unknown;

		if (
			typeof json !== "object" ||
			json === null ||
			!("txid" in json) ||
			typeof (json as { txid: unknown }).txid !== "number"
		) {
			throw new Error(
				"Update API must return JSON with a numeric txid for Electric reconciliation.",
			);
		}

		return json as { txid: number };
	});
