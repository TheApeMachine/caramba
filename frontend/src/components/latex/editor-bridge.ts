import { Store } from "@tanstack/store";
import type { SetBlockKindOptions } from "./blocks/convert-block";
import type {
	HeadingLevel,
	PaperBlock,
	PaperBlockKind,
	PaperMetadata,
} from "./model/types";

export type EditorBridgeAPI = {
	getBlocks: () => PaperBlock[];
	getMetadata: () => PaperMetadata;
	updateText: (id: string, text: string) => void;
	updateLatex: (id: string, latex: string) => void;
	insertParagraphAfter: (afterId: string, text?: string) => string;
	insertHeadingAfter: (afterId: string, level: HeadingLevel) => string;
	insertEquationAfter: (afterId: string, latex?: string) => string;
	insertListAfter: (afterId: string, ordered: boolean) => string;
	insertBlockAfter: (afterId: string, block: PaperBlock) => string;
	removeBlock: (id: string) => void;
	reorderBlock: (
		sourceId: string,
		targetId: string,
		position: "above" | "below",
	) => void;
	setBlockKind: (
		id: string,
		kind: PaperBlockKind,
		options?: SetBlockKindOptions,
	) => void;
	updateMetadata: (patch: Partial<PaperMetadata>) => void;
	scrollToBlock: (id: string) => void;
};

/*
editorBridgeStore exposes the active paper editor's API to assistant
tools through a Tanstack Store. The editor publishes its API as part
of its provider render; tools read the latest value with editorBridge.get()
or subscribe via the store. No imperative lifecycle, no useEffect.
*/
const editorBridgeStore = new Store<EditorBridgeAPI | null>(null);

export const editorBridge = {
	publish(api: EditorBridgeAPI | null): void {
		editorBridgeStore.setState(() => api);
	},
	get(): EditorBridgeAPI | null {
		return editorBridgeStore.state;
	},
	store: editorBridgeStore,
};
