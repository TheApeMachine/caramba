import { auth } from "@clerk/tanstack-react-start/server";
import { createFileRoute } from "@tanstack/react-router";
import { backendBaseURL } from "#/lib/backend-http";

export const Route = createFileRoute("/api/compute/optimizer")({
	server: {
		handlers: {
			GET: async ({ request }) => {
				let authorizationHeader = request.headers.get("Authorization");

				if (!authorizationHeader?.startsWith("Bearer ")) {
					const authResult = await auth();
					const token = await authResult.getToken();

					if (token !== null && token.length > 0) {
						authorizationHeader = `Bearer ${token}`;
					}
				}

				const backendHeaders =
					authorizationHeader !== null && authorizationHeader.length > 0
						? { Authorization: authorizationHeader }
						: undefined;

				const response = await fetch(
					`${backendBaseURL()}/backend/compute/optimizer`,
					{ headers: backendHeaders },
				);
				const data = await response.json();

				return new Response(JSON.stringify(data), {
					status: response.status,
					headers: { "Content-Type": "application/json" },
				});
			},
		},
	},
});
