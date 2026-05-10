import { createServerFn } from "@tanstack/react-start";
import pg from "pg";
import { researchProjectRowSchema } from "#/lib/research-project-schema";

let researchProjectsPool: pg.Pool | undefined;

function getPool(): pg.Pool | null {
	const connectionString = process.env.DATABASE_URL;
	if (!connectionString) {
		return null;
	}
	if (!researchProjectsPool) {
		researchProjectsPool = new pg.Pool({ connectionString });
	}
	return researchProjectsPool;
}

/**
 * Insert into Postgres and return a txid Electric can match on the shape stream.
 * Same pattern as TanStack's Electric todo example (pg_current_xact_id()::xid).
 */
async function insertResearchProjectRow(
	data: ReturnType<typeof researchProjectRowSchema.parse>,
): Promise<{ txid: number }> {
	const pool = getPool();
	if (!pool) {
		throw new Error(
			"Set DATABASE_URL (e.g. postgresql://postgres:postgres@127.0.0.1:54321/electric) for local inserts, or RESEARCH_PROJECT_INSERT_URL for an HTTP API that returns { txid: number }.",
		);
	}

	const client = await pool.connect();
	try {
		await client.query("BEGIN");
		await client.query(
			`INSERT INTO research_projects (id, name, description, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5)`,
			[data.id, data.name, data.description, data.created_at, data.updated_at],
		);
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
}

async function insertResearchProjectViaHttp(
	url: string,
	data: ReturnType<typeof researchProjectRowSchema.parse>,
): Promise<{ txid: number }> {
	const res = await fetch(url, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			...data,
			created_at: data.created_at.toISOString(),
			updated_at: data.updated_at.toISOString(),
		}),
	});

	if (!res.ok) {
		const text = await res.text();
		throw new Error(`Research project insert failed (${res.status}): ${text}`);
	}

	const json: unknown = await res.json();
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
}

/**
 * Persists a research project for Electric reconciliation:
 * - If `RESEARCH_PROJECT_INSERT_URL` is set, POSTs JSON to that URL (expects `{ txid: number }`).
 * - Else uses `DATABASE_URL` and inserts directly into `research_projects`.
 */
export const createResearchProject = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => researchProjectRowSchema.parse(data))
	.handler(async ({ data }) => {
		const url = process.env.RESEARCH_PROJECT_INSERT_URL;
		if (url) {
			return insertResearchProjectViaHttp(url, data);
		}
		return insertResearchProjectRow(data);
	});
