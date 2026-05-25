import { useMemo } from "react";
import { usePublishAssistantContext } from "#/components/assistant/use-publish-assistant-context";
import { usePaperEditor } from "#/components/latex/context";
import type { PaperBlock } from "#/components/latex/model/types";

const focusedBlockSummary = (block: PaperBlock | undefined): string | null => {
	if (!block) {
		return null;
	}

	if (block.type === "equation") {
		return JSON.stringify(
			{
				id: block.id,
				type: block.type,
				display: block.display,
				label: block.label ?? null,
				latex: block.latex,
			},
			null,
			2,
		);
	}

	if (block.type === "heading") {
		return JSON.stringify(
			{
				id: block.id,
				type: block.type,
				level: block.level,
				text: block.text,
				presentation: block.presentation ?? null,
			},
			null,
			2,
		);
	}

	if (block.type === "paragraph") {
		return JSON.stringify(
			{
				id: block.id,
				type: block.type,
				text: block.text,
			},
			null,
			2,
		);
	}

	return JSON.stringify(
		{
			id: block.id,
			type: block.type,
			ordered: block.ordered,
			text: block.text,
		},
		null,
		2,
	);
};

/*
PaperAssistantContext publishes semantic paper state to the assistant bridge,
including the focused block AST regardless of panel visibility.
*/
export const PaperAssistantContext = () => {
	const { blocks, focusedBlockId } = usePaperEditor();
	const focusedBlock = blocks.find((block) => block.id === focusedBlockId);

	const focusedEntry = useMemo(() => {
		const summary = focusedBlockSummary(focusedBlock);

		if (!summary) {
			return null;
		}

		return {
			key: "paper_focus",
			label: "Focused paper block",
			value: summary,
			persistent: true,
		};
	}, [focusedBlock]);

	usePublishAssistantContext(focusedEntry);

	return null;
};
