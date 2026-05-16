import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import {
	type Collection,
	createCollection,
	localStorageCollectionOptions,
} from "@tanstack/react-db";
import { z } from "zod";

export const AssistantSessionPersona = z.object({
	session_id: z.uuid(),
	persona_id: z.uuid(),
	position: z.number().int().default(0),
});

export type AssistantSessionPersonaRow = z.infer<typeof AssistantSessionPersona>;

const shapeUrl =
	typeof window !== "undefined"
		? `${window.location.origin}/api/shape/assistant-session-personas`
		: "/api/shape/assistant-session-personas";

function compositeKey(item: AssistantSessionPersonaRow): string {
	return `${item.session_id}:${item.persona_id}`;
}

let cloud: Collection<AssistantSessionPersonaRow> | null = null;
let local: Collection<AssistantSessionPersonaRow> | null = null;

function buildCloud() {
	return createCollection(
		electricCollectionOptions({
			id: "assistant_session_personas",
			schema: AssistantSessionPersona,
			getKey: compositeKey,
			shapeOptions: { url: shapeUrl },
		}),
	);
}

function buildLocal() {
	return createCollection(
		localStorageCollectionOptions({
			id: "assistant_session_personas_local",
			storageKey: "caramba:assistant:session_personas",
			schema: AssistantSessionPersona,
			getKey: compositeKey,
		}),
	);
}

export function getSessionPersonasCollection(mode: "cloud" | "local") {
	if (mode === "local") {
		if (!local) local = buildLocal();
		return local;
	}

	if (!cloud) cloud = buildCloud();
	return cloud;
}
