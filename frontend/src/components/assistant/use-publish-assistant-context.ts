import { useEffect } from "react";
import {
	assistantContextBridge,
	type SemanticContextEntry,
} from "./assistant-context-bridge";

/*
usePublishAssistantContext publishes logical assistant context that survives
cosmetic UI changes such as collapsed panels or hidden elements.
*/
export const usePublishAssistantContext = (
	entry: SemanticContextEntry | null,
): void => {
	useEffect(() => {
		if (!entry) {
			return;
		}

		assistantContextBridge.publish(entry);

		return () => {
			assistantContextBridge.unpublish(entry.key);
		};
	}, [entry]);
};
