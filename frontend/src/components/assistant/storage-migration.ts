import type { Collection } from "@tanstack/react-db";
import type { AssistantMessageRow } from "#/collections/assistant_messages";
import type { AssistantPersonaRow } from "#/collections/assistant_personas";
import type { AssistantSessionPersonaRow } from "#/collections/assistant_session_personas";
import type { AssistantSessionRow } from "#/collections/assistant_sessions";
import { DEFAULT_PERSONA, DEFAULT_WINDOW_SIZE } from "./types";

const LEGACY_SESSIONS_KEY = "caramba:assistant:sessions";
const MIGRATED_FLAG = "caramba:assistant:migrated";

type LegacyMessage = {
	id: string;
	role: "system" | "user" | "assistant";
	parts: unknown[];
	createdAt?: string | number | Date;
	personaId?: string;
	personaName?: string;
};

type LegacyPersona = {
	id: string;
	name: string;
	systemPrompt: string;
	model: string;
	temperature?: number;
	maxTokens?: number;
};

type LegacySession = {
	id: string;
	title: string;
	createdAt: number;
	messages: LegacyMessage[];
	personas: LegacyPersona[];
	windowSize?: number;
};

interface MigrationStores {
	personas: Collection<AssistantPersonaRow>;
	sessions: Collection<AssistantSessionRow>;
	messages: Collection<AssistantMessageRow>;
	sessionPersonas: Collection<AssistantSessionPersonaRow>;
}

/*
importLocalStorageIfNeeded performs a one-shot import of legacy session data
from the old localStorage layout into the active collection set. Runs once
per browser and gates on a MIGRATED_FLAG so re-mounts are no-ops.
*/
export function importLocalStorageIfNeeded(stores: MigrationStores) {
	if (typeof window === "undefined") return;
	if (window.localStorage.getItem(MIGRATED_FLAG) === "1") return;

	const raw = window.localStorage.getItem(LEGACY_SESSIONS_KEY);

	if (!raw) {
		window.localStorage.setItem(MIGRATED_FLAG, "1");
		return;
	}

	let legacy: LegacySession[] = [];

	try {
		legacy = JSON.parse(raw) as LegacySession[];
	} catch {
		window.localStorage.setItem(MIGRATED_FLAG, "1");
		return;
	}

	(async () => {
		const seenPersonaIds = new Set<string>();

		for (const session of legacy) {
			const sessionId = session.id;
			const personaIds: string[] = [];

			for (const persona of session.personas ?? []) {
				const personaId = persona.id;

				if (!seenPersonaIds.has(personaId) && !stores.personas.get(personaId)) {
					await stores.personas.insert({
						id: personaId,
						scope: "personal",
						owner_id: null,
						organization_slug: null,
						name: persona.name,
						system_prompt: persona.systemPrompt ?? "",
						model: persona.model ?? DEFAULT_PERSONA.model,
						temperature: persona.temperature ?? DEFAULT_PERSONA.temperature,
						max_tokens: persona.maxTokens ?? DEFAULT_PERSONA.maxTokens,
						adapter_type: "openai",
						endpoint_url: null,
						created_at: new Date(),
						updated_at: new Date(),
					});
					seenPersonaIds.add(personaId);
				}

				personaIds.push(personaId);
			}

			if (!stores.sessions.get(sessionId)) {
				await stores.sessions.insert(
					{
						id: sessionId,
						scope: "personal",
						owner_id: null,
						organization_slug: null,
						title: session.title,
						window_size: session.windowSize ?? DEFAULT_WINDOW_SIZE,
						created_at: new Date(session.createdAt ?? Date.now()),
						updated_at: new Date(session.createdAt ?? Date.now()),
					},
					{ metadata: { personaIds } },
				);
			}

			for (const [position, personaId] of personaIds.entries()) {
				const key = `${sessionId}:${personaId}`;
				if (stores.sessionPersonas.get(key)) continue;
				await stores.sessionPersonas.insert({
					session_id: sessionId,
					persona_id: personaId,
					position,
				});
			}

			for (const message of session.messages ?? []) {
				if (stores.messages.get(message.id)) continue;
				await stores.messages.insert({
					id: message.id,
					session_id: sessionId,
					role: message.role,
					parts: message.parts,
					persona_id: message.personaId ?? null,
					persona_name: message.personaName ?? null,
					created_at: new Date(message.createdAt ?? Date.now()),
				});
			}
		}

		window.localStorage.setItem(MIGRATED_FLAG, "1");
	})().catch((error) => {
		console.error("[assistant] legacy import failed:", error);
	});
}
