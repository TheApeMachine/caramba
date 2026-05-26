import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import { createCollection } from "@tanstack/react-db";
import { z } from "zod";
import { shapeUrl } from "#/lib/electric-shape";
import { createTeam } from "#/server/create-team";
import { updateTeam } from "#/server/update-team";

export const Team = z.object({
	id: z.uuid(),
	organization_slug: z.string().min(1),
	name: z.string().min(1),
	// Slug is authoritative on the server (it derives one from name and
	// resolves uniqueness collisions), so the client can send "" on insert
	// and Electric will replace the optimistic row once the shape syncs.
	slug: z.string().default(""),
	description: z.string().default(""),
	color: z.string().default(""),
	emoji: z.string().default(""),
	privacy_mode: z.enum(["shared", "local"]).default("shared"),
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
		onUpdate: async ({ transaction }) => {
			if (!transaction.mutations.length) {
				throw new Error("onUpdate called with no mutations");
			}

			const mutation = transaction.mutations[0];
			const next = mutation.modified;
			const previous = mutation.original;

			const payload: Record<string, unknown> = { id: next.id };

			if (next.name !== previous.name) payload.name = next.name;
			if (next.description !== previous.description)
				payload.description = next.description;
			if (next.color !== previous.color) payload.color = next.color;
			if (next.emoji !== previous.emoji) payload.emoji = next.emoji;
			if (next.privacy_mode !== previous.privacy_mode)
				payload.privacy_mode = next.privacy_mode;

			const result = await updateTeam({ data: payload as never });

			if (import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true") {
				return;
			}

			if (!result?.txid) {
				throw new Error("updateTeam returned no txid");
			}

			return { timeout: 60_000, txid: result.txid };
		},
	}),
);
