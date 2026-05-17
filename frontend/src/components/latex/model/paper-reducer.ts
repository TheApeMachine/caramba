import type {
	HeadingLevel,
	PaperBlock,
	PaperBlockKind,
} from "#/components/latex/model/types";

export type SetBlockKindOptions = {
	level?: HeadingLevel;
	ordered?: boolean;
};

export type PaperAction =
	| { type: "UPDATE_TEXT"; id: string; text: string }
	| { type: "UPDATE_LATEX"; id: string; latex: string }
	| { type: "INSERT_AFTER"; afterId: string; block: PaperBlock }
	| { type: "REMOVE_BLOCK"; id: string }
	| {
			type: "REORDER_BLOCK";
			sourceId: string;
			targetId: string;
			position: "above" | "below";
	  }
	| {
			type: "SET_BLOCK_KIND";
			id: string;
			kind: PaperBlockKind;
			options?: SetBlockKindOptions;
	  };

function newBlockId(): string {
	return crypto.randomUUID();
}

function readText(block: PaperBlock): string {
	if (block.type === "equation") {
		return block.latex;
	}

	return block.text;
}

function convertBlock(
	block: PaperBlock,
	kind: PaperBlockKind,
	options: SetBlockKindOptions = {},
): PaperBlock {
	if (kind === block.type) {
		return block;
	}

	const carriedText = readText(block);

	if (kind === "heading") {
		return {
			id: block.id,
			type: "heading",
			level: options.level ?? 2,
			text: carriedText,
		};
	}

	if (kind === "equation") {
		return {
			id: block.id,
			type: "equation",
			latex: carriedText,
			display: true,
		};
	}

	if (kind === "list") {
		return {
			id: block.id,
			type: "list",
			ordered: options.ordered ?? false,
			text: carriedText,
		};
	}

	return { id: block.id, type: "paragraph", text: carriedText };
}

function reorderBlock(
	blocks: PaperBlock[],
	sourceId: string,
	targetId: string,
	position: "above" | "below",
): PaperBlock[] {
	if (sourceId === targetId) {
		return blocks;
	}

	const sourceIndex = blocks.findIndex((block) => block.id === sourceId);
	const targetIndex = blocks.findIndex((block) => block.id === targetId);

	if (sourceIndex === -1 || targetIndex === -1) {
		return blocks;
	}

	const without = blocks.slice();
	const [moved] = without.splice(sourceIndex, 1);

	const adjustedTarget =
		sourceIndex < targetIndex ? targetIndex - 1 : targetIndex;
	const insertAt = position === "below" ? adjustedTarget + 1 : adjustedTarget;

	without.splice(insertAt, 0, moved);

	return without;
}

export function paperReducer(
	blocks: PaperBlock[],
	action: PaperAction,
): PaperBlock[] {
	switch (action.type) {
		case "UPDATE_TEXT":
			return blocks.map((block) =>
				block.id === action.id && block.type !== "equation"
					? { ...block, text: action.text }
					: block,
			);

		case "UPDATE_LATEX":
			return blocks.map((block) =>
				block.id === action.id && block.type === "equation"
					? { ...block, latex: action.latex }
					: block,
			);

		case "INSERT_AFTER": {
			const index = blocks.findIndex((block) => block.id === action.afterId);

			if (index === -1) {
				return blocks;
			}

			return [
				...blocks.slice(0, index + 1),
				action.block,
				...blocks.slice(index + 1),
			];
		}

		case "REMOVE_BLOCK": {
			if (blocks.length <= 1) {
				return blocks;
			}

			const filtered = blocks.filter((block) => block.id !== action.id);

			if (filtered.length === 0) {
				return [{ id: newBlockId(), type: "paragraph", text: "" }];
			}

			return filtered;
		}

		case "REORDER_BLOCK":
			return reorderBlock(
				blocks,
				action.sourceId,
				action.targetId,
				action.position,
			);

		case "SET_BLOCK_KIND":
			return blocks.map((block) =>
				block.id === action.id
					? convertBlock(block, action.kind, action.options)
					: block,
			);

		default: {
			const exhaustive: never = action;
			return exhaustive;
		}
	}
}

export function createInitialPaperBlocks(): PaperBlock[] {
	return [
		{
			id: "initial-heading",
			type: "heading",
			level: 1,
			text: "Untitled paper",
		},
		{ id: "initial-paragraph", type: "paragraph", text: "" },
	];
}
