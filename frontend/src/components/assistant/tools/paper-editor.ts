import { toolDefinition } from "@tanstack/ai";
import { editorBridge } from "#/components/latex/editor-bridge";

/*
paper_list_blocks — read current paper state.
Agents must call this first to get block IDs before inserting or editing.
*/
export const paperListBlocks = toolDefinition({
	name: "paper_list_blocks",
	description: "Returns the current list of blocks in the paper with their IDs, types, and content. Call this first before inserting or editing blocks.",
	inputSchema: {},
}).client(() => {
	const api = editorBridge.get();
	if (!api) return { error: "Paper editor is not open." };

	const blocks = api.getBlocks();
	const meta = api.getMetadata();

	return {
		metadata: meta,
		blocks: blocks.map((b) => {
			if (b.type === "heading") return { id: b.id, type: "heading", level: b.level, text: b.text };
			if (b.type === "paragraph") return { id: b.id, type: "paragraph", text: b.text };
			return { id: b.id, type: "equation", latex: b.latex, label: b.label, display: b.display };
		}),
	};
});

/*
paper_update_metadata — set title, authors, abstract, keywords.
*/
export const paperUpdateMetadata = toolDefinition({
	name: "paper_update_metadata",
	description: "Updates the paper metadata (title, authors, abstract, keywords). Pass only the fields you want to change.",
	inputSchema: {
		type: "object" as const,
		properties: {
			title:    { type: "string" },
			authors:  { type: "string", description: "One author per line." },
			abstract: { type: "string" },
			keywords: { type: "string", description: "Comma-separated." },
		},
	},
}).client((args: unknown) => {
	const typed = args as { title?: string; authors?: string; abstract?: string; keywords?: string };
	const api = editorBridge.get();
	if (!api) return { error: "Paper editor is not open." };
	api.updateMetadata(typed);
	return { ok: true };
});

/*
paper_insert_block — insert a new block after a given block ID.
Use afterId: "last" to append at the end.
*/
export const paperInsertBlock = toolDefinition({
	name: "paper_insert_block",
	description: "Inserts a new block (paragraph, heading, or equation) after the given block ID. Use afterId 'last' to append at end.",
	inputSchema: {
		type: "object" as const,
		required: ["afterId", "blockType"],
		properties: {
			afterId:   { type: "string", description: "ID of the block to insert after, or 'last'." },
			blockType: { type: "string", enum: ["paragraph", "heading", "equation"] },
			text:      { type: "string", description: "Text content for paragraph or heading blocks." },
			level:     { type: "number", enum: [1, 2, 3], description: "Heading level (1–3). Required for heading blocks." },
			latex:     { type: "string", description: "LaTeX source for equation blocks." },
		},
	},
}).client((args: unknown) => {
	const typed = args as {
		afterId: string;
		blockType: "paragraph" | "heading" | "equation";
		text?: string;
		level?: 1 | 2 | 3;
		latex?: string;
	};
	const api = editorBridge.get();
	if (!api) return { error: "Paper editor is not open." };

	const blocks = api.getBlocks();
	const afterId = typed.afterId === "last" ? (blocks.at(-1)?.id ?? blocks[0].id) : typed.afterId;

	let newId: string;

	if (typed.blockType === "heading") {
		newId = api.insertHeadingAfter(afterId, typed.level ?? 2);
		if (typed.text) api.updateText(newId, typed.text);
	} else if (typed.blockType === "equation") {
		newId = api.insertEquationAfter(afterId, typed.latex ?? "");
	} else {
		newId = api.insertParagraphAfter(afterId, typed.text ?? "");
	}

	api.scrollToBlock(newId);
	return { ok: true, newBlockId: newId };
});

/*
paper_update_block — overwrite the content of an existing block.
*/
export const paperUpdateBlock = toolDefinition({
	name: "paper_update_block",
	description: "Updates the content of an existing block by its ID.",
	inputSchema: {
		type: "object" as const,
		required: ["id"],
		properties: {
			id:    { type: "string" },
			text:  { type: "string", description: "New text for paragraph or heading blocks." },
			latex: { type: "string", description: "New LaTeX for equation blocks." },
		},
	},
}).client((args: unknown) => {
	const typed = args as { id: string; text?: string; latex?: string };
	const api = editorBridge.get();
	if (!api) return { error: "Paper editor is not open." };

	if (typed.text !== undefined) api.updateText(typed.id, typed.text);
	if (typed.latex !== undefined) api.updateLatex(typed.id, typed.latex);

	api.scrollToBlock(typed.id);
	return { ok: true };
});

/*
paper_remove_block — delete a block by ID.
*/
export const paperRemoveBlock = toolDefinition({
	name: "paper_remove_block",
	description: "Removes a block from the paper by its ID.",
	inputSchema: {
		type: "object" as const,
		required: ["id"],
		properties: {
			id: { type: "string" },
		},
	},
}).client((args: unknown) => {
	const typed = args as { id: string };
	const api = editorBridge.get();
	if (!api) return { error: "Paper editor is not open." };
	api.removeBlock(typed.id);
	return { ok: true };
});

/*
paper_scroll_to_block — scroll the user's view to a specific block.
Useful for the guided-tour / attention-directing capability.
*/
export const paperScrollToBlock = toolDefinition({
	name: "paper_scroll_to_block",
	description: "Scrolls the paper editor to bring a specific block into view. Use this to direct the user's attention.",
	inputSchema: {
		type: "object" as const,
		required: ["id"],
		properties: {
			id: { type: "string" },
		},
	},
}).client((args: unknown) => {
	const typed = args as { id: string };
	const api = editorBridge.get();
	if (!api) return { error: "Paper editor is not open." };
	api.scrollToBlock(typed.id);
	return { ok: true };
});

export const paperEditorTools = [
	paperListBlocks,
	paperUpdateMetadata,
	paperInsertBlock,
	paperUpdateBlock,
	paperRemoveBlock,
	paperScrollToBlock,
];
