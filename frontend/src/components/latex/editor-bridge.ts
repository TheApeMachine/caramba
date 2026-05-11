import type { HeadingLevel, PaperBlock, PaperMetadata } from "./model/types";

/*
EditorBridge is a singleton that the PaperEditorProvider registers its methods
into when it mounts, and clears when it unmounts. Agent tools call into this
bridge so they can manipulate the editor from outside the React tree.
*/
export type EditorBridgeAPI = {
	getBlocks: () => PaperBlock[];
	getMetadata: () => PaperMetadata;
	updateText: (id: string, text: string) => void;
	updateLatex: (id: string, latex: string) => void;
	insertParagraphAfter: (afterId: string, text?: string) => string;
	insertHeadingAfter: (afterId: string, level: HeadingLevel) => string;
	insertEquationAfter: (afterId: string, latex?: string) => string;
	removeBlock: (id: string) => void;
	setBlockKind: (id: string, kind: "paragraph" | "heading" | "equation", level?: HeadingLevel) => void;
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
