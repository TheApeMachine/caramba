import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import {
	createCollection,
	localStorageCollectionOptions,
} from "@tanstack/react-db";
import { z } from "zod";
import { shapeUrl } from "#/lib/electric-shape";

export const AssistantSessionPersona = z.object({
	session_id: z.uuid(),
	persona_id: z.uuid(),
	position: z.number().int().default(0),
});

export type AssistantSessionPersonaRow = z.infer<
	typeof AssistantSessionPersona
>;

const compositeKey = (item: AssistantSessionPersonaRow): string =>
	`${item.session_id}:${item.persona_id}`;

export const assistantSessionPersonasCollection = createCollection(
	electricCollectionOptions({
		id: "assistant_session_personas",
		schema: AssistantSessionPersona,
		getKey: compositeKey,
		shapeOptions: {
			url: shapeUrl("assistant-session-personas"),
		},
	}),
);

export const assistantSessionPersonasLocalCollection = createCollection(
	localStorageCollectionOptions({
		id: "assistant_session_personas_local",
		storageKey: "caramba:assistant:session_personas",
		schema: AssistantSessionPersona,
		getKey: compositeKey,
	}),
);
