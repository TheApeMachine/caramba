import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import {
	createCollection,
	localStorageCollectionOptions,
} from "@tanstack/react-db";
import { z } from "zod";
import { electricAwaitOptions, shapeUrl } from "#/lib/electric-shape";
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

export const assistantMessagesCollection = createCollection(
	electricCollectionOptions({
		id: "assistant_messages",
		schema: AssistantMessage,
		getKey: (item) => item.id,
		shapeOptions: {
			url: shapeUrl("assistant-messages"),
			parser: { timestamptz: (value: string) => new Date(value) },
		},
		onInsert: async ({ transaction }) => {
			const row = transaction.mutations[0].modified;
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

			return electricAwaitOptions(result?.txid);
		},
	}),
);

export const assistantMessagesLocalCollection = createCollection(
	localStorageCollectionOptions({
		id: "assistant_messages_local",
		storageKey: "caramba:assistant:messages",
		schema: AssistantMessage,
		getKey: (item) => item.id,
	}),
);
