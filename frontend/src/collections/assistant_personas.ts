import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import {
	type Collection,
	createCollection,
	localStorageCollectionOptions,
} from "@tanstack/react-db";
import { z } from "zod";
import {
	createPersona,
	deletePersona,
	updatePersona,
} from "#/server/assistant-personas";

export const AssistantPersona = z.object({
	id: z.uuid(),
	scope: z.enum(["global", "team", "personal"]),
	owner_id: z.string().nullable().optional(),
	organization_slug: z.string().nullable().optional(),
	name: z.string().min(1),
	system_prompt: z.string().default(""),
	model: z.string().min(1),
	temperature: z.number().default(0.7),
	max_tokens: z.number().int().default(2048),
	adapter_type: z.enum(["openai", "ollama", "openai-compat"]).default("openai"),
	endpoint_url: z.string().nullable().optional(),
	created_at: z.coerce.date(),
	updated_at: z.coerce.date(),
});

export type AssistantPersonaRow = z.infer<typeof AssistantPersona>;

const shapeUrl =
	typeof window !== "undefined"
		? `${window.location.origin}/api/shape/assistant-personas`
		: "/api/shape/assistant-personas";

const skipTxidAwait =
	import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true";

function awaitOptions(txid: number | undefined) {
	if (skipTxidAwait || typeof txid !== "number") return undefined;
	return { timeout: 60_000, txid };
}

let cloud: Collection<AssistantPersonaRow> | null = null;
let local: Collection<AssistantPersonaRow> | null = null;

function buildCloud() {
	return createCollection(
		electricCollectionOptions({
			id: "assistant_personas",
			schema: AssistantPersona,
			getKey: (item) => item.id,
			shapeOptions: {
				url: shapeUrl,
				parser: { timestamptz: (value: string) => new Date(value) },
			},
			onInsert: async ({ transaction }) => {
				const row = transaction.mutations[0].modified as AssistantPersonaRow;
				const result = await createPersona({
					data: {
						id: row.id,
						scope: row.scope,
						name: row.name,
						system_prompt: row.system_prompt,
						model: row.model,
						temperature: row.temperature,
						max_tokens: row.max_tokens,
						adapter_type: row.adapter_type,
						endpoint_url: row.endpoint_url ?? "",
					},
				});

				return awaitOptions(result?.txid);
			},
			onUpdate: async ({ transaction }) => {
				const row = transaction.mutations[0].modified as AssistantPersonaRow;
				const result = await updatePersona({
					data: {
						id: row.id,
						scope: row.scope,
						name: row.name,
						system_prompt: row.system_prompt,
						model: row.model,
						temperature: row.temperature,
						max_tokens: row.max_tokens,
						adapter_type: row.adapter_type,
						endpoint_url: row.endpoint_url ?? "",
					},
				});

				return awaitOptions(result?.txid);
			},
			onDelete: async ({ transaction }) => {
				const row = transaction.mutations[0].original as AssistantPersonaRow;
				const result = await deletePersona({
					data: { id: row.id, scope: row.scope },
				});

				return awaitOptions(result?.txid);
			},
		}),
	);
}

function buildLocal() {
	return createCollection(
		localStorageCollectionOptions({
			id: "assistant_personas_local",
			storageKey: "caramba:assistant:personas",
			schema: AssistantPersona,
			getKey: (item) => item.id,
		}),
	);
}

export function getPersonasCollection(mode: "cloud" | "local") {
	if (mode === "local") {
		if (!local) local = buildLocal();
		return local;
	}

	if (!cloud) cloud = buildCloud();
	return cloud;
}
