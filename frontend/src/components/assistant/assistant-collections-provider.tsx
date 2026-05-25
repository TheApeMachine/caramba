import { createContext, type ReactNode, useContext, useMemo } from "react";
import { getMessagesCollection } from "#/collections/assistant_messages";
import { getPersonasCollection } from "#/collections/assistant_personas";
import { getSessionPersonasCollection } from "#/collections/assistant_session_personas";
import { getSessionsCollection } from "#/collections/assistant_sessions";
import type { SyncMode } from "#/lib/electric-shape";
import { useAssistantMode } from "./use-assistant-mode";

// biome-ignore lint/suspicious/noExplicitAny: TanStack collection union widens too far
export type AssistantCollection = any;

export type AssistantCollections = {
	mode: SyncMode;
	personas: AssistantCollection;
	sessions: AssistantCollection;
	messages: AssistantCollection;
	sessionPersonas: AssistantCollection;
};

const AssistantCollectionsContext = createContext<AssistantCollections | null>(
	null,
);

/*
AssistantCollectionsProvider resolves the active sync transport once at the
assistant subtree root. UI hooks consume collections without repeating
local/cloud branching.
*/
export const AssistantCollectionsProvider = ({
	children,
}: {
	children: ReactNode;
}) => {
	const { mode } = useAssistantMode();

	const collections = useMemo<AssistantCollections>(
		() => ({
			mode,
			personas: getPersonasCollection(mode),
			sessions: getSessionsCollection(mode),
			messages: getMessagesCollection(mode),
			sessionPersonas: getSessionPersonasCollection(mode),
		}),
		[mode],
	);

	return (
		<AssistantCollectionsContext.Provider value={collections}>
			{children}
		</AssistantCollectionsContext.Provider>
	);
};

export const useAssistantCollections = (): AssistantCollections => {
	const collections = useContext(AssistantCollectionsContext);

	if (!collections) {
		throw new Error(
			"useAssistantCollections must be used within AssistantCollectionsProvider",
		);
	}

	return collections;
};
