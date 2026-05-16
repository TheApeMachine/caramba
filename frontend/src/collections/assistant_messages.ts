import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import {
	type Collection,
	createCollection,
	localStorageCollectionOptions,
} from "@tanstack/react-db";
import { z } from "zod";
import { createMessage } from "#/server/assistant-sessions";

export const AssistantMessage = z.object({
	id: z.uuid(),
	session_id: z.uuid(),
	role: z.enum(["system", "user", "assistant"]),
	parts: z.any().default([]),
	persona_id: z.string().nullable().optional(),
	persona_name: z.string().nullable().optional(),
	created_at: z.coerce.date(),
});

export type AssistantMessageRow = z.infer<typeof AssistantMessage>;

const shapeUrl =
	typeof window !== "undefined"
		? `${window.location.origin}/api/shape/assistant-messages`
		: "/api/shape/assistant-messages";

const skipTxidAwait =
	import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true";

function awaitOptions(txid: number | undefined) {
	if (skipTxidAwait || typeof txid !== "number") return undefined;
	return { timeout: 60_000, txid };
}

let cloud: Collection<AssistantMessageRow> | null = null;
let local: Collection<AssistantMessageRow> | null = null;

function buildCloud() {
	return createCollection(
		electricCollectionOptions({
			id: "assistant_messages",
			schema: AssistantMessage,
			getKey: (item) => item.id,
			shapeOptions: {
				url: shapeUrl,
				parser: { timestamptz: (value: string) => new Date(value) },
			},
			onInsert: async ({ transaction }) => {
				const row = transaction.mutations[0].modified as AssistantMessageRow;
				const result = await createMessage({
					data: {
						id: row.id,
						session_id: row.session_id,
						role: row.role,
						parts: Array.isArray(row.parts) ? row.parts : [],
						persona_id: row.persona_id ?? "",
						persona_name: row.persona_name ?? "",
					},
				});

				return awaitOptions(result?.txid);
			},
		}),
	);
}

function buildLocal() {
	return createCollection(
		localStorageCollectionOptions({
			id: "assistant_messages_local",
			storageKey: "caramba:assistant:messages",
			schema: AssistantMessage,
			getKey: (item) => item.id,
		}),
	);
}

export function getMessagesCollection(mode: "cloud" | "local") {
	if (mode === "local") {
		if (!local) local = buildLocal();
		return local;
	}

	if (!cloud) cloud = buildCloud();
	return cloud;
}
