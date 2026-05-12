import { auth } from "@clerk/tanstack-react-start/server";
import { createFileRoute } from "@tanstack/react-router";

const ELECTRIC_URL =
	process.env.ELECTRIC_URL ?? "http://127.0.0.1:3010/v1/shape";

export const Route = createFileRoute("/api/shape/research-projects")({
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

				const orgSlug = authState.orgSlug ?? authState.userId;

				const incomingUrl = new URL(request.url);
				const upstreamUrl = new URL(ELECTRIC_URL);

				for (const [key, value] of incomingUrl.searchParams.entries()) {
					upstreamUrl.searchParams.set(key, value);
				}

				upstreamUrl.searchParams.set("table", "research_projects");
				upstreamUrl.searchParams.set("where", "organization_slug = $1");
				upstreamUrl.searchParams.set("params[1]", orgSlug);

				const upstreamResponse = await fetch(upstreamUrl.toString(), {
					headers: { Accept: "application/json" },
					signal: request.signal,
				});

				return new Response(upstreamResponse.body, {
					status: upstreamResponse.status,
					headers: {
						"Content-Type":
							upstreamResponse.headers.get("Content-Type") ??
							"application/json",
						"Cache-Control":
							upstreamResponse.headers.get("Cache-Control") ?? "no-store",
						"Electric-Cursor":
							upstreamResponse.headers.get("Electric-Cursor") ?? "",
						"Electric-Handle":
							upstreamResponse.headers.get("Electric-Handle") ?? "",
						"Electric-Schema":
							upstreamResponse.headers.get("Electric-Schema") ?? "",
						"Electric-Offset":
							upstreamResponse.headers.get("Electric-Offset") ?? "",
					},
				});
			},
		},
	},
});
