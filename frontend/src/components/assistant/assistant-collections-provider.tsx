import { createContext, type ReactNode, useContext, useMemo } from "react";
import {
	assistantMessagesCollection,
	assistantMessagesLocalCollection,
} from "#/collections/assistant_messages";
import {
	assistantPersonasCollection,
	assistantPersonasLocalCollection,
} from "#/collections/assistant_personas";
import {
	assistantSessionPersonasCollection,
	assistantSessionPersonasLocalCollection,
} from "#/collections/assistant_session_personas";
import {
	assistantSessionsCollection,
	assistantSessionsLocalCollection,
} from "#/collections/assistant_sessions";
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
	const isLocal = mode === "local";

	const collections = useMemo<AssistantCollections>(
		() => ({
			mode,
			personas: isLocal
				? assistantPersonasLocalCollection
				: assistantPersonasCollection,
			sessions: isLocal
				? assistantSessionsLocalCollection
				: assistantSessionsCollection,
			messages: isLocal
				? assistantMessagesLocalCollection
				: assistantMessagesCollection,
			sessionPersonas: isLocal
				? assistantSessionPersonasLocalCollection
				: assistantSessionPersonasCollection,
		}),
		[isLocal, mode],
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
