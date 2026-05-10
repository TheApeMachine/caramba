"use client";

import { usePaperEditor } from "#/components/latex/context";
import { BlockRow } from "#/components/latex/panels/block-row";
import { Flex } from "#/components/ui/flex";
import { ScrollArea } from "#/components/ui/scroll-area";

export function WritingCanvas() {
	const { blocks } = usePaperEditor();

	return (
		<ScrollArea className="min-h-0 flex-1" scrollFade scrollbarGutter>
			<Flex.Column
				className="mx-auto max-w-3xl px-4 py-8 sm:px-8 sm:py-12"
				fullWidth
				gap={1}
			>
				{blocks.map((block) => (
					<BlockRow block={block} key={block.id} />
				))}
			</Flex.Column>
		</ScrollArea>
	);
}
