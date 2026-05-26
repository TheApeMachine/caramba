import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import { createCollection } from "@tanstack/react-db";
import { z } from "zod";
import { shapeUrl } from "#/lib/electric-shape";
import { createTeam } from "#/server/create-team";

export const Team = z.object({
	id: z.uuid(),
	organization_slug: z.string().min(1),
	name: z.string().min(1),
	slug: z.string().min(1),
	description: z.string().default(""),
	created_at: z.coerce.date(),
	updated_at: z.coerce.date(),
});

export type TeamRow = z.infer<typeof Team>;

export const teamCollection = createCollection(
	electricCollectionOptions({
		id: "teams",
		schema: Team,
		getKey: (item) => item.id,
		shapeOptions: {
			url: shapeUrl("teams"),
			parser: {
				timestamptz: (value: string) => new Date(value),
			},
		},
		onInsert: async ({ transaction }) => {
			if (!transaction.mutations.length) {
				throw new Error("onInsert called with no mutations");
			}

			const row = transaction.mutations[0].modified;

			const result = await createTeam({ data: row });

			if (import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true") {
				return;
			}

			if (!result?.txid) {
				throw new Error("createTeam returned no txid");
			}

			return { timeout: 60_000, txid: result.txid };
		},
	}),
);
