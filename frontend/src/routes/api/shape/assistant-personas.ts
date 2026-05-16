import { auth } from "@clerk/tanstack-react-start/server";
import { createFileRoute } from "@tanstack/react-router";

const ELECTRIC_URL =
	process.env.ELECTRIC_URL ?? "http://127.0.0.1:3010/v1/shape";

export const Route = createFileRoute("/api/shape/assistant-personas")({
	server: {
		handlers: {
			GET: async ({ request }) => {
				const authState = await auth();

				if (!authState.userId) {
					return new Response(JSON.stringify({ error: "Unauthorized" }), {
						status: 401,
						headers: { "Content-Type": "application/json" },
					});
				}

				const orgSlug = authState.orgSlug ?? "";
				const subject = authState.userId;

				const incoming = new URL(request.url);
				const upstream = new URL(ELECTRIC_URL);

				for (const [key, value] of incoming.searchParams.entries()) {
					upstream.searchParams.set(key, value);
				}

				upstream.searchParams.set("table", "assistant_personas");
				upstream.searchParams.set(
					"where",
					`scope='global' OR (scope='team' AND organization_slug=$1) OR (scope='personal' AND owner_id=$2)`,
				);
				upstream.searchParams.set("params[1]", orgSlug);
				upstream.searchParams.set("params[2]", subject);

				const upstreamResponse = await fetch(upstream.toString(), {
					headers: { Accept: "application/json" },
					signal: request.signal,
				});

				return new Response(upstreamResponse.body, {
					status: upstreamResponse.status,
					headers: passthroughHeaders(upstreamResponse),
				});
			},
		},
	},
});

function passthroughHeaders(response: Response): HeadersInit {
	return {
		"Content-Type":
			response.headers.get("Content-Type") ?? "application/json",
		"Cache-Control": response.headers.get("Cache-Control") ?? "no-store",
		"Electric-Cursor": response.headers.get("Electric-Cursor") ?? "",
		"Electric-Handle": response.headers.get("Electric-Handle") ?? "",
		"Electric-Schema": response.headers.get("Electric-Schema") ?? "",
		"Electric-Offset": response.headers.get("Electric-Offset") ?? "",
	};
}
