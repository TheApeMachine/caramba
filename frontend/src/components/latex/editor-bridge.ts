import type { SetBlockKindOptions } from "./model/paper-reducer";
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

let api: EditorBridgeAPI | null = null;

export const editorBridge = {
	register(impl: EditorBridgeAPI) {
		api = impl;
	},
	unregister() {
		api = null;
	},
	get(): EditorBridgeAPI | null {
		return api;
	},
};
