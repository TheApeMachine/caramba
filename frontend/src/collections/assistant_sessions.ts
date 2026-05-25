import { z } from "zod";
import {
	createDualModeCollection,
	electricAwaitOptions,
} from "#/lib/dual-mode-collection";
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

type SessionMutationContext = {
	personaIds?: string[];
};

export const getSessionsCollection = createDualModeCollection({
	cacheKey: "assistant_sessions",
	schema: AssistantSession,
	getKey: (item) => item.id,
	cloud: {
		id: "assistant_sessions",
		shapePath: "assistant-sessions",
		parser: { timestamptz: (value: string) => new Date(value) },
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

			return electricAwaitOptions(result?.txid);
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

			return electricAwaitOptions(result?.txid);
		},
		onDelete: async ({ transaction }) => {
			const row = transaction.mutations[0].original as AssistantSessionRow;
			const result = await deleteSession({ data: { id: row.id } });

			return electricAwaitOptions(result?.txid);
		},
	},
	local: {
		id: "assistant_sessions_local",
		storageKey: "caramba:assistant:sessions",
	},
});
