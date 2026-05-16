import { useLiveQuery } from "@tanstack/react-db";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
	type AssistantMessageRow,
	getMessagesCollection,
} from "#/collections/assistant_messages";
import {
	type AssistantPersonaRow,
	getPersonasCollection,
} from "#/collections/assistant_personas";
import {
	type AssistantSessionPersonaRow,
	getSessionPersonasCollection,
} from "#/collections/assistant_session_personas";
import {
	type AssistantSessionRow,
	getSessionsCollection,
} from "#/collections/assistant_sessions";
import { importLocalStorageIfNeeded } from "./storage-migration";
import {
	type AdapterType,
	DEFAULT_PERSONA,
	DEFAULT_WINDOW_SIZE,
	type Persona,
	type PersonaScope,
	type Session,
	type SessionScope,
	type UIMessage,
} from "./types";
import { useAssistantMode } from "./use-assistant-mode";

const ACTIVE_KEY = "caramba:assistant:active";

function readActiveId(): string | null {
	if (typeof window === "undefined") return null;
	return window.localStorage.getItem(ACTIVE_KEY);
}

function writeActiveId(id: string): void {
	if (typeof window === "undefined") return;
	window.localStorage.setItem(ACTIVE_KEY, id);
}

function personaFromRow(row: AssistantPersonaRow): Persona {
	return {
		id: row.id,
		scope: row.scope as PersonaScope,
		name: row.name,
		systemPrompt: row.system_prompt,
		model: row.model,
		temperature: row.temperature,
		maxTokens: row.max_tokens,
		adapterType: row.adapter_type as AdapterType,
		endpointUrl: row.endpoint_url ?? "",
	};
}

function rowFromPersona(persona: Persona): AssistantPersonaRow {
	const now = new Date();
	return {
		id: persona.id,
		scope: persona.scope,
		owner_id: null,
		organization_slug: null,
		name: persona.name,
		system_prompt: persona.systemPrompt,
		model: persona.model,
		temperature: persona.temperature,
		max_tokens: persona.maxTokens,
		adapter_type: persona.adapterType,
		endpoint_url: persona.endpointUrl,
		created_at: now,
		updated_at: now,
	};
}

function messageFromRow(row: AssistantMessageRow): UIMessage {
	return {
		id: row.id,
		role: row.role,
		parts: (row.parts as UIMessage["parts"]) ?? [],
		createdAt: row.created_at,
		personaId: row.persona_id ?? undefined,
		personaName: row.persona_name ?? undefined,
	};
}

function rowFromMessage(
	sessionId: string,
	message: UIMessage,
): AssistantMessageRow {
	return {
		id: message.id,
		session_id: sessionId,
		role: message.role,
		parts: message.parts,
		persona_id: message.personaId ?? null,
		persona_name: message.personaName ?? null,
		created_at: message.createdAt ?? new Date(),
	};
}

function materializeSession(
	row: AssistantSessionRow,
	allMessages: AssistantMessageRow[],
	allJoins: AssistantSessionPersonaRow[],
	allPersonas: AssistantPersonaRow[],
): Session {
	const joins = allJoins
		.filter((join) => join.session_id === row.id)
		.sort((a, b) => a.position - b.position);

	const personas = joins
		.map((join) => allPersonas.find((p) => p.id === join.persona_id))
		.filter((value): value is AssistantPersonaRow => Boolean(value))
		.map(personaFromRow);

	const messages = allMessages
		.filter((message) => message.session_id === row.id)
		.sort((a, b) => a.created_at.getTime() - b.created_at.getTime())
		.map(messageFromRow);

	return {
		id: row.id,
		scope: row.scope as SessionScope,
		title: row.title,
		createdAt: row.created_at.getTime(),
		messages,
		personas:
			personas.length > 0 ? personas : [{ ...DEFAULT_PERSONA }],
		windowSize: row.window_size,
	};
}

function newSessionRow(scope: SessionScope = "personal"): AssistantSessionRow {
	const now = new Date();
	return {
		id: crypto.randomUUID(),
		scope,
		owner_id: null,
		organization_slug: null,
		title: "New conversation",
		window_size: DEFAULT_WINDOW_SIZE,
		created_at: now,
		updated_at: now,
	};
}

async function replaceSessionPersonas(
	joinCollection: ReturnType<typeof getSessionPersonasCollection>,
	sessionId: string,
	personaIds: string[],
	existing: AssistantSessionPersonaRow[],
) {
	for (const join of existing) {
		if (join.session_id !== sessionId) continue;
		await joinCollection.delete(`${join.session_id}:${join.persona_id}`);
	}

	for (const [position, personaId] of personaIds.entries()) {
		await joinCollection.insert({
			session_id: sessionId,
			persona_id: personaId,
			position,
		});
	}
}

/*
useSession is the single entry point for the assistant UI. It joins sessions,
messages, personas, and the session_personas join into Session[] regardless of
whether the active collections are Electric-backed (cloud) or
localStorage-backed (local). Mode switches recreate the active collections
through the `get*Collection(mode)` factories.
*/
export function useSession() {
	const { mode } = useAssistantMode();

	const personasCollection = getPersonasCollection(mode);
	const sessionsCollection = getSessionsCollection(mode);
	const messagesCollection = getMessagesCollection(mode);
	const sessionPersonasCollection = getSessionPersonasCollection(mode);

	const importedRef = useRef<string | null>(null);

	useEffect(() => {
		if (importedRef.current === mode) return;
		importedRef.current = mode;
		importLocalStorageIfNeeded({
			personas: personasCollection,
			sessions: sessionsCollection,
			messages: messagesCollection,
			sessionPersonas: sessionPersonasCollection,
		});
	}, [mode, personasCollection, sessionsCollection, messagesCollection, sessionPersonasCollection]);

	const personasQuery = useLiveQuery(
		(query) => query.from({ row: personasCollection }),
		[personasCollection],
	);
	const sessionsQuery = useLiveQuery(
		(query) => query.from({ row: sessionsCollection }),
		[sessionsCollection],
	);
	const messagesQuery = useLiveQuery(
		(query) => query.from({ row: messagesCollection }),
		[messagesCollection],
	);
	const joinsQuery = useLiveQuery(
		(query) => query.from({ row: sessionPersonasCollection }),
		[sessionPersonasCollection],
	);

	const personas = (personasQuery.data as AssistantPersonaRow[] | undefined) ?? [];
	const sessions = (sessionsQuery.data as AssistantSessionRow[] | undefined) ?? [];
	const messages = (messagesQuery.data as AssistantMessageRow[] | undefined) ?? [];
	const joins = (joinsQuery.data as AssistantSessionPersonaRow[] | undefined) ?? [];

	const materializedSessions = useMemo(
		() => sessions.map((row) => materializeSession(row, messages, joins, personas)),
		[sessions, messages, joins, personas],
	);

	const [activeId, setActiveId] = useState<string | null>(() => readActiveId());

	useEffect(() => {
		if (materializedSessions.length === 0 && activeId !== null) {
			setActiveId(null);
			return;
		}

		if (activeId && materializedSessions.some((session) => session.id === activeId)) return;

		const fallback = materializedSessions[0]?.id ?? null;

		if (fallback) {
			writeActiveId(fallback);
			setActiveId(fallback);
		}
	}, [materializedSessions, activeId]);

	const session = useMemo<Session>(() => {
		const found = materializedSessions.find((entry) => entry.id === activeId);

		if (found) return found;

		return {
			id: "pending",
			scope: "personal",
			title: "New conversation",
			createdAt: Date.now(),
			messages: [],
			personas: [{ ...DEFAULT_PERSONA }],
			windowSize: DEFAULT_WINDOW_SIZE,
		};
	}, [materializedSessions, activeId]);

	const setActive = useCallback((id: string) => {
		writeActiveId(id);
		setActiveId(id);
	}, []);

	const createSession = useCallback(async () => {
		const row = newSessionRow();
		const personaIds =
			personas.length > 0
				? personas.map((persona) => persona.id)
				: [DEFAULT_PERSONA.id];

		await sessionsCollection.insert(row, { metadata: { personaIds } });

		for (const [position, personaId] of personaIds.entries()) {
			await sessionPersonasCollection.insert({
				session_id: row.id,
				persona_id: personaId,
				position,
			});
		}

		writeActiveId(row.id);
		setActiveId(row.id);
	}, [personas, sessionsCollection, sessionPersonasCollection]);

	const deleteSession = useCallback(
		async (id: string) => {
			await sessionsCollection.delete(id);

			for (const join of joins) {
				if (join.session_id !== id) continue;
				await sessionPersonasCollection.delete(`${join.session_id}:${join.persona_id}`);
			}

			for (const message of messages) {
				if (message.session_id !== id) continue;
				await messagesCollection.delete(message.id);
			}
		},
		[sessionsCollection, sessionPersonasCollection, messagesCollection, joins, messages],
	);

	const appendMessages = useCallback(
		async (incoming: UIMessage[]) => {
			if (!session.id || session.id === "pending") return;

			for (const message of incoming) {
				await messagesCollection.insert(rowFromMessage(session.id, message));
			}
		},
		[session.id, messagesCollection],
	);

	const upsertMessage = useCallback(
		async (message: UIMessage) => {
			if (!session.id || session.id === "pending") return;

			const row = rowFromMessage(session.id, message);
			const existing = messagesCollection.get(message.id);

			if (existing) {
				await messagesCollection.update(message.id, (draft) => {
					Object.assign(draft, row);
				});
				return;
			}

			await messagesCollection.insert(row);
		},
		[session.id, messagesCollection],
	);

	const updatePersona = useCallback(
		async (persona: Persona) => {
			const row = rowFromPersona(persona);

			if (personasCollection.get(persona.id)) {
				await personasCollection.update(persona.id, (draft) => {
					Object.assign(draft, row);
				});
				return;
			}

			await personasCollection.insert(row);
		},
		[personasCollection],
	);

	const addPersona = useCallback(
		async (persona: Persona) => {
			await personasCollection.insert(rowFromPersona(persona));

			if (session.id && session.id !== "pending") {
				const personaIds = [
					...session.personas.map((entry) => entry.id),
					persona.id,
				];
				await replaceSessionPersonas(
					sessionPersonasCollection,
					session.id,
					personaIds,
					joins,
				);
			}
		},
		[personasCollection, sessionPersonasCollection, session.id, session.personas, joins],
	);

	const removePersona = useCallback(
		async (personaId: string) => {
			if (!session.id || session.id === "pending") return;

			const personaIds = session.personas
				.map((entry) => entry.id)
				.filter((id) => id !== personaId);

			if (personaIds.length === 0) return;

			await replaceSessionPersonas(
				sessionPersonasCollection,
				session.id,
				personaIds,
				joins,
			);
		},
		[sessionPersonasCollection, session.id, session.personas, joins],
	);

	const setWindowSize = useCallback(
		async (size: number) => {
			if (!session.id || session.id === "pending") return;

			await sessionsCollection.update(
				session.id,
				{ metadata: { personaIds: session.personas.map((entry) => entry.id) } },
				(draft) => {
					draft.window_size = size;
				},
			);
		},
		[sessionsCollection, session.id, session.personas],
	);

	return {
		sessions: materializedSessions,
		session,
		setActive,
		createSession,
		deleteSession,
		appendMessages,
		upsertMessage,
		updatePersona,
		addPersona,
		removePersona,
		setWindowSize,
	};
}
