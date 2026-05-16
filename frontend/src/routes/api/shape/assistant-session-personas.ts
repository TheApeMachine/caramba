import { auth } from "@clerk/tanstack-react-start/server";
import { createFileRoute } from "@tanstack/react-router";
import pg from "pg";

const ELECTRIC_URL =
	process.env.ELECTRIC_URL ?? "http://127.0.0.1:3010/v1/shape";

let assistantShapePool: pg.Pool | undefined;

function getPool(): pg.Pool | null {
	const connectionString = process.env.DATABASE_URL;
	if (!connectionString) return null;
	if (!assistantShapePool) {
		assistantShapePool = new pg.Pool({ connectionString });
	}
	return assistantShapePool;
}

async function visibleSessionIds(
	subject: string,
	orgSlug: string,
): Promise<string[]> {
	const pool = getPool();
	if (!pool) {
		throw new Error(
			"Set DATABASE_URL so the assistant shape proxy can resolve visible sessions.",
		);
	}

	const result = await pool.query<{ id: string }>(
		`SELECT id FROM assistant_sessions
		  WHERE (scope = 'personal' AND owner_id = $1)
		     OR (scope = 'team'     AND organization_slug = $2)`,
		[subject, orgSlug],
	);

	return result.rows.map((row) => row.id);
}

function buildInClause(ids: string[]): {
	where: string;
	params: Record<string, string>;
} {
	const placeholders = ids.map((_, index) => `$${index + 1}`).join(", ");
	const params: Record<string, string> = {};

	for (const [index, id] of ids.entries()) {
		params[`params[${index + 1}]`] = id;
	}

	return { where: `session_id IN (${placeholders})`, params };
}

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

				const sessionIds = await visibleSessionIds(subject, orgSlug);

				const incoming = new URL(request.url);
				const upstream = new URL(ELECTRIC_URL);

				for (const [key, value] of incoming.searchParams.entries()) {
					upstream.searchParams.set(key, value);
				}

				upstream.searchParams.set("table", "assistant_session_personas");

				if (sessionIds.length === 0) {
					upstream.searchParams.set("where", "session_id IS NULL");
				} else {
					const { where, params } = buildInClause(sessionIds);
					upstream.searchParams.set("where", where);

					for (const [key, value] of Object.entries(params)) {
						upstream.searchParams.set(key, value);
					}
				}

				const upstreamResponse = await fetch(upstream.toString(), {
					headers: { Accept: "application/json" },
					signal: request.signal,
				});

				return new Response(upstreamResponse.body, {
					status: upstreamResponse.status,
					headers: passthroughElectricHeaders(upstreamResponse),
				});
			},
		},
	},
});

function passthroughElectricHeaders(response: Response): HeadersInit {
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
