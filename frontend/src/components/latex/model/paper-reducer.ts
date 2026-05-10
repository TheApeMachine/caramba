import type { HeadingLevel, PaperBlock } from "#/components/latex/model/types";

export type PaperAction =
	| { type: "UPDATE_TEXT"; id: string; text: string }
	| { type: "INSERT_AFTER"; afterId: string; block: PaperBlock }
	| { type: "REMOVE_BLOCK"; id: string }
	| {
			type: "SET_BLOCK_KIND";
			id: string;
			kind: "paragraph" | "heading";
			level?: HeadingLevel;
	  };

function newParagraphId(): string {
	return crypto.randomUUID();
}

export function paperReducer(
	blocks: PaperBlock[],
	action: PaperAction,
): PaperBlock[] {
	switch (action.type) {
		case "UPDATE_TEXT":
			return blocks.map((b) =>
				b.id === action.id ? { ...b, text: action.text } : b,
			);
		case "INSERT_AFTER": {
			const i = blocks.findIndex((b) => b.id === action.afterId);
			if (i === -1) {
				return blocks;
			}
			return [...blocks.slice(0, i + 1), action.block, ...blocks.slice(i + 1)];
		}
		case "REMOVE_BLOCK": {
			if (blocks.length <= 1) {
				return blocks;
			}
			const filtered = blocks.filter((b) => b.id !== action.id);
			if (filtered.length === 0) {
				return [{ id: newParagraphId(), type: "paragraph", text: "" }];
			}
			return filtered;
		}
		case "SET_BLOCK_KIND":
			return blocks.map((b) => {
				if (b.id !== action.id) {
					return b;
				}
				if (action.kind === "heading") {
					const level: HeadingLevel = action.level ?? 2;
					return {
						id: b.id,
						type: "heading",
						level,
						text: b.text,
					};
				}
				return {
					id: b.id,
					type: "paragraph",
					text: b.text,
				};
			});
		default: {
			const _exhaustive: never = action;
			return _exhaustive;
		}
	}
}

export function createInitialPaperBlocks(): PaperBlock[] {
	return [
		{
			id: crypto.randomUUID(),
			type: "heading",
			level: 1,
			text: "Untitled paper",
		},
		{ id: crypto.randomUUID(), type: "paragraph", text: "" },
	];
}
