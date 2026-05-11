"use client";

import { FunctionSquareIcon, HeadingIcon, PlusIcon } from "lucide-react";
import { usePaperEditor } from "#/components/latex/context";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";

export function PaperEditorToolbar() {
	const { blocks, insertHeadingAfter, insertParagraphAfter, insertEquationAfter, focusedBlockId } =
		usePaperEditor();

	const anchorId = focusedBlockId ?? blocks[blocks.length - 1]?.id ?? blocks[0]?.id ?? null;

	const addHeading = () => { if (anchorId) insertHeadingAfter(anchorId, 2); };
	const addParagraph = () => { if (anchorId) insertParagraphAfter(anchorId, ""); };
	const addEquation = () => { if (anchorId) insertEquationAfter(anchorId); };

	return (
		<Flex.Row
			align="center"
			className="shrink-0 border-border border-b bg-background/80 backdrop-blur-sm"
			justify="end"
			padding={2}
			gap={2}
		>
			<Button size="sm" type="button" variant="ghost" onClick={addHeading}>
				<HeadingIcon />
				Section
			</Button>
			<Button size="sm" type="button" variant="ghost" onClick={addParagraph}>
				<PlusIcon />
				Paragraph
			</Button>
			<Button size="sm" type="button" variant="ghost" onClick={addEquation}>
				<FunctionSquareIcon />
				Equation
			</Button>
		</Flex.Row>
	);
}
