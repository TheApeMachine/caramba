import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import pg from "pg";
import { z } from "zod";
import { insertKanbanCardRowWithClient } from "#/server/kanban-cards";

let featureRequestPool: pg.Pool | undefined;

function getPool(): pg.Pool | null {
	const connectionString = process.env.DATABASE_URL;
	if (!connectionString) {
		return null;
	}
	if (!featureRequestPool) {
		featureRequestPool = new pg.Pool({ connectionString });
	}
	return featureRequestPool;
}

async function resolveFeatureRequestsProjectId(
	client: pg.PoolClient,
): Promise<string> {
	const envProjectId = process.env.FEATURE_REQUEST_PROJECT_ID?.trim();
	if (envProjectId) {
		return envProjectId;
	}

	const organizationSlug =
		process.env.FEATURE_REQUEST_ORGANIZATION_SLUG?.trim() ?? "caramba";

	const lookup = await client.query<{ id: string }>(
		`SELECT id FROM research_projects
     WHERE organization_slug = $1 AND project_slug = $2
     LIMIT 1`,
		[organizationSlug, "requests"],
	);

	const projectId = lookup.rows[0]?.id;
	if (projectId === undefined) {
		throw new Error(
			"Feature request project is not configured: seed research_projects (caramba/requests) or set FEATURE_REQUEST_PROJECT_ID.",
		);
	}

	return projectId;
}

async function nextBacklogSortOrder(
	client: pg.PoolClient,
	researchProjectId: string,
): Promise<number> {
	const res = await client.query<{ next: string }>(
		`SELECT (COALESCE(MAX(sort_order), -1) + 1)::text AS next
     FROM kanban_cards
     WHERE research_project_id = $1 AND column_key = 'backlog'`,
		[researchProjectId],
	);
	const raw = res.rows[0]?.next;
	if (raw === undefined) {
		return 0;
	}
	return Number.parseInt(raw, 10);
}

const submitFeatureRequestInputSchema = z.object({
	title: z.string().min(3).max(200),
	description: z.string().min(1).max(8000),
	contact_email: z.string().email().optional(),
});

/*
submitFeatureRequest inserts a backlog card on the canonical "requests" research project.
Requires a Clerk session (same gate as /request-feature client route).
*/
export const submitFeatureRequest = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) =>
		submitFeatureRequestInputSchema.parse(data),
	)
	.handler(async ({ data }) => {
		const authenticationState = await auth();

		if (!authenticationState.userId) {
			throw new Error("Feature requests require a signed-in account.");
		}

		const pool = getPool();
		if (!pool) {
			throw new Error(
				"Set DATABASE_URL so feature requests can be persisted to Postgres.",
			);
		}

		const client = await pool.connect();
		try {
			await client.query("BEGIN");

			const researchProjectId = await resolveFeatureRequestsProjectId(client);
			const sortOrder = await nextBacklogSortOrder(client, researchProjectId);
			const id = crypto.randomUUID();
			const now = new Date();

			await insertKanbanCardRowWithClient(client, {
				id,
				research_project_id: researchProjectId,
				column_key: "backlog",
				sort_order: sortOrder,
				title: data.title.trim(),
				description: data.description.trim(),
				priority: "medium",
				labels_json: "[]",
				assignees_json: "[]",
				due_date: null,
				requested_by: data.contact_email?.trim() ?? null,
				created_at: now,
				updated_at: now,
			});

			const txidRes = await client.query<{ txid: string }>(
				"SELECT pg_current_xact_id()::xid::text AS txid",
			);
			const txidRaw = txidRes.rows[0]?.txid;
			if (txidRaw === undefined) {
				throw new Error("Failed to get transaction ID");
			}
			const txid = Number.parseInt(txidRaw, 10);

			await client.query("COMMIT");
			return { txid };
		} catch (err) {
			try {
				await client.query("ROLLBACK");
			} catch {
				/* ignore */
			}
			throw err;
		} finally {
			client.release();
		}
	});
