import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import {
	type Collection,
	createCollection,
	localStorageCollectionOptions,
} from "@tanstack/react-db";
import { z } from "zod";
import {
	createSession,
	deleteSession,
	updateSession,
} from "#/server/assistant-sessions";

export const AssistantSession = z.object({
	id: z.uuid(),
	scope: z.enum(["team", "personal"]),
	owner_id: z.string().nullable().optional(),
	organization_slug: z.string().nullable().optional(),
	title: z.string().default("New conversation"),
	window_size: z.number().int().default(20),
	created_at: z.coerce.date(),
	updated_at: z.coerce.date(),
});

export type AssistantSessionRow = z.infer<typeof AssistantSession>;

const shapeUrl =
	typeof window !== "undefined"
		? `${window.location.origin}/api/shape/assistant-sessions`
		: "/api/shape/assistant-sessions";

const skipTxidAwait =
	import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true";

function awaitOptions(txid: number | undefined) {
	if (skipTxidAwait || typeof txid !== "number") return undefined;
	return { timeout: 60_000, txid };
}

type SessionMutationContext = {
	personaIds?: string[];
};

let cloud: Collection<AssistantSessionRow> | null = null;
let local: Collection<AssistantSessionRow> | null = null;

function buildCloud() {
	return createCollection(
		electricCollectionOptions({
			id: "assistant_sessions",
			schema: AssistantSession,
			getKey: (item) => item.id,
			shapeOptions: {
				url: shapeUrl,
				parser: { timestamptz: (value: string) => new Date(value) },
			},
			onInsert: async ({ transaction }) => {
				const row = transaction.mutations[0].modified as AssistantSessionRow;
				const meta = (transaction.metadata ?? {}) as SessionMutationContext;
				const result = await createSession({
					data: {
						id: row.id,
						scope: row.scope,
						title: row.title,
						window_size: row.window_size,
						persona_ids: meta.personaIds ?? [],
					},
				});

				return awaitOptions(result?.txid);
			},
			onUpdate: async ({ transaction }) => {
				const row = transaction.mutations[0].modified as AssistantSessionRow;
				const meta = (transaction.metadata ?? {}) as SessionMutationContext;
				const result = await updateSession({
					data: {
						id: row.id,
						scope: row.scope,
						title: row.title,
						window_size: row.window_size,
						persona_ids: meta.personaIds ?? [],
					},
				});

				return awaitOptions(result?.txid);
			},
			onDelete: async ({ transaction }) => {
				const row = transaction.mutations[0].original as AssistantSessionRow;
				const result = await deleteSession({ data: { id: row.id } });

				return awaitOptions(result?.txid);
			},
		}),
	);
}

function buildLocal() {
	return createCollection(
		localStorageCollectionOptions({
			id: "assistant_sessions_local",
			storageKey: "caramba:assistant:sessions",
			schema: AssistantSession,
			getKey: (item) => item.id,
		}),
	);
}

export function getSessionsCollection(mode: "cloud" | "local") {
	if (mode === "local") {
		if (!local) local = buildLocal();
		return local;
	}

	if (!cloud) cloud = buildCloud();
	return cloud;
}
