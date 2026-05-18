import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import {
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

const shapeUrl = () => {
	if (typeof window === "undefined") return "http://localhost/api/shape/assistant-session-personas";
	return `${window.location.origin}/api/shape/assistant-session-personas`;
}

const compositeKey = (item: AssistantSessionPersonaRow): string => {
	return `${item.session_id}:${item.persona_id}`;
}

let cloud: ReturnType<typeof buildCloud> | null = null;
let local: ReturnType<typeof buildLocal> | null = null;

const buildCloud = () => {
	return createCollection(
		electricCollectionOptions({
			id: "assistant_session_personas",
			schema: AssistantSessionPersona,
			getKey: compositeKey,
			shapeOptions: { url: shapeUrl() },
		}),
	);
}

const buildLocal = () => {
	return createCollection(
		localStorageCollectionOptions({
			id: "assistant_session_personas_local",
			storageKey: "caramba:assistant:session_personas",
			schema: AssistantSessionPersona,
			getKey: compositeKey,
		}),
	);
};

export const getSessionPersonasCollection = (mode: "cloud" | "local") => {
	if (mode === "local") {
		local ??= buildLocal();
		return local;
	}

	cloud ??= buildCloud();
	return cloud;
}
