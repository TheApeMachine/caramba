import { auth } from "@clerk/tanstack-react-start/server";
import { createFileRoute } from "@tanstack/react-router";

const ELECTRIC_URL =
	process.env.ELECTRIC_URL ?? "http://127.0.0.1:3010/v1/shape";

export const Route = createFileRoute("/api/shape/assistant-session-personas")({
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

				upstream.searchParams.set("table", "assistant_session_personas");
				upstream.searchParams.set(
					"where",
					`session_id IN (
                        SELECT id FROM assistant_sessions
                         WHERE (scope='personal' AND owner_id=$1)
                            OR (scope='team' AND organization_slug=$2)
                    )`,
				);
				upstream.searchParams.set("params[1]", subject);
				upstream.searchParams.set("params[2]", orgSlug);

				const upstreamResponse = await fetch(upstream.toString(), {
					headers: { Accept: "application/json" },
					signal: request.signal,
				});

				return new Response(upstreamResponse.body, {
					status: upstreamResponse.status,
					headers: {
						"Content-Type":
							upstreamResponse.headers.get("Content-Type") ?? "application/json",
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
