"use client";

import { HeadingIcon, PlusIcon } from "lucide-react";
import { usePaperEditor } from "#/components/latex/context";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

export function PaperEditorToolbar() {
	const { blocks, insertHeadingAfter, insertParagraphAfter, focusedBlockId } =
		usePaperEditor();

	const anchorAfterId =
		focusedBlockId ?? blocks[blocks.length - 1]?.id ?? blocks[0]?.id ?? null;

	const addHeading = () => {
		if (!anchorAfterId) {
			return;
		}
		insertHeadingAfter(anchorAfterId, 2);
	};

	const addParagraph = () => {
		if (!anchorAfterId) {
			return;
		}
		insertParagraphAfter(anchorAfterId, "");
	};

	return (
		<Flex.Row
			align="center"
			className="shrink-0 border-border border-b bg-background/80 backdrop-blur-sm"
			justify="between"
			padding={3}
			gap={2}
			wrap="wrap"
		>
			<Typography.Small variant="muted">
				Editing locally — connect storage when ready.
			</Typography.Small>
			<Flex.Row align="center" gap={2} wrap="wrap">
				<Button size="sm" type="button" variant="outline" onClick={addHeading}>
					<HeadingIcon />
					Section (H2)
				</Button>
				<Button
					size="sm"
					type="button"
					variant="outline"
					onClick={addParagraph}
				>
					<PlusIcon />
					Paragraph
				</Button>
			</Flex.Row>
		</Flex.Row>
	);
}
