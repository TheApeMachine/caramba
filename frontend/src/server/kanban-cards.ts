import { createServerFn } from "@tanstack/react-start";
import pg from "pg";
import { z } from "zod";
import {
	kanbanCardRowSchema,
	kanbanColumnKeySchema,
} from "#/lib/kanban-card-schema";

let kanbanCardsPool: pg.Pool | undefined;

function getPool(): pg.Pool | null {
	const connectionString = process.env.DATABASE_URL;
	if (!connectionString) {
		return null;
	}
	if (!kanbanCardsPool) {
		kanbanCardsPool = new pg.Pool({ connectionString });
	}
	return kanbanCardsPool;
}

async function transactionTxid(
	callback: (client: pg.PoolClient) => Promise<void>,
): Promise<{ txid: number }> {
	const pool = getPool();
	if (!pool) {
		throw new Error(
			"Set DATABASE_URL for Kanban persistence (same Postgres Electric uses locally).",
		);
	}

	const client = await pool.connect();
	try {
		await client.query("BEGIN");
		await callback(client);
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

/*
insertKanbanCardRow inserts one Kanban row inside an outer transaction callback-friendly helper.
*/
export async function insertKanbanCardRowWithClient(
	client: pg.PoolClient,
	data: z.infer<typeof kanbanCardRowSchema>,
): Promise<void> {
	await client.query(
		`INSERT INTO kanban_cards (
      id,
      research_project_id,
      column_key,
      sort_order,
      title,
      description,
      priority,
      labels_json,
      assignees_json,
      due_date,
      requested_by,
      created_at,
      updated_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)`,
		[
			data.id,
			data.research_project_id,
			data.column_key,
			data.sort_order,
			data.title,
			data.description,
			data.priority,
			data.labels_json,
			data.assignees_json,
			data.due_date,
			data.requested_by,
			data.created_at,
			data.updated_at,
		],
	);
}

const syncOrderingInputSchema = z.object({
	updates: z.array(
		z.object({
			id: z.uuid(),
			column_key: kanbanColumnKeySchema,
			sort_order: z.number().int(),
		}),
	),
});

const patchKanbanCardInputSchema = z.object({
	id: z.uuid(),
	title: z.string().optional(),
	description: z.string().optional(),
	priority: z.enum(["low", "medium", "high", "critical"]).optional(),
	labels_json: z.string().optional(),
	assignees_json: z.string().optional(),
	due_date: z.coerce.date().nullable().optional(),
});

const deleteKanbanCardInputSchema = z.object({
	id: z.uuid(),
});

/*
insertKanbanCard persists a Kanban row and returns an Electric txid.
*/
export const insertKanbanCard = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => kanbanCardRowSchema.parse(data))
	.handler(async ({ data }) => {
		return transactionTxid(async (client) => {
			await insertKanbanCardRowWithClient(client, data);
		});
	});

/*
syncKanbanOrdering applies column and order updates in a single transaction.
*/
export const syncKanbanOrdering = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => syncOrderingInputSchema.parse(data))
	.handler(async ({ data }) => {
		return transactionTxid(async (client) => {
			for (const row of data.updates) {
				await client.query(
					`UPDATE kanban_cards
           SET column_key = $2, sort_order = $3, updated_at = NOW()
           WHERE id = $1`,
					[row.id, row.column_key, row.sort_order],
				);
			}
		});
	});

/*
patchKanbanCard updates editable card fields.
*/
export const patchKanbanCard = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => patchKanbanCardInputSchema.parse(data))
	.handler(async ({ data }) => {
		return transactionTxid(async (client) => {
			const sets: string[] = ["updated_at = NOW()"];
			const values: unknown[] = [];
			let index = 1;

			const pushSet = (fragment: string, value: unknown) => {
				sets.push(`${fragment} = $${index}`);
				values.push(value);
				index++;
			};

			if (data.title !== undefined) {
				pushSet("title", data.title);
			}
			if (data.description !== undefined) {
				pushSet("description", data.description);
			}
			if (data.priority !== undefined) {
				pushSet("priority", data.priority);
			}
			if (data.labels_json !== undefined) {
				pushSet("labels_json", data.labels_json);
			}
			if (data.assignees_json !== undefined) {
				pushSet("assignees_json", data.assignees_json);
			}
			if (data.due_date !== undefined) {
				pushSet("due_date", data.due_date);
			}

			values.push(data.id);

			await client.query(
				`UPDATE kanban_cards SET ${sets.join(", ")} WHERE id = $${index}`,
				values,
			);
		});
	});

/*
deleteKanbanCard removes a card row.
*/
export const deleteKanbanCard = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => deleteKanbanCardInputSchema.parse(data))
	.handler(async ({ data }) => {
		return transactionTxid(async (client) => {
			await client.query(`DELETE FROM kanban_cards WHERE id = $1`, [data.id]);
		});
	});
