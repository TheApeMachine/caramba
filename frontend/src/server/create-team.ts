import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { Team } from "#/collections/team";
import { backendBaseURL } from "#/lib/backend-http";

export const createTeam = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => Team.parse(data))
	.handler(async ({ data }) => {
		const authenticationState = await auth();
		const token = await authenticationState.getToken();

		if (!authenticationState.userId || !token) {
			throw new Error("Team writes require a signed-in account.");
		}

		const response = await fetch(`${backendBaseURL()}/backend/teams`, {
			method: "POST",
			headers: {
				Authorization: `Bearer ${token}`,
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				id: data.id,
				name: data.name,
				slug: data.slug ?? "",
				description: data.description ?? "",
			}),
		});

		if (!response.ok) {
			const text = await response.text();
			throw new Error(`Team insert failed (${response.status}): ${text}`);
		}

		const json = (await response.json()) as unknown;

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
