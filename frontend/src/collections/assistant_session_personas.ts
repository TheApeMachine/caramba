import { z } from "zod";
import { createDualModeCollection } from "#/lib/dual-mode-collection";

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

export const getSessionPersonasCollection = createDualModeCollection({
	cacheKey: "assistant_session_personas",
	schema: AssistantSessionPersona,
	getKey: compositeKey,
	cloud: {
		id: "assistant_session_personas",
		shapePath: "assistant-session-personas",
	},
	local: {
		id: "assistant_session_personas_local",
		storageKey: "caramba:assistant:session_personas",
	},
});
