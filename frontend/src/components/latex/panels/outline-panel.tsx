"use client";

import { ListTreeIcon } from "lucide-react";
import { usePaperEditor } from "#/components/latex/context";
import type { PaperBlock } from "#/components/latex/model/types";
import { Flex } from "#/components/ui/flex";
import { ScrollArea } from "#/components/ui/scroll-area";
import { Typography } from "#/components/ui/typography";

function outlineEntries(
	blocks: PaperBlock[],
): { id: string; label: string; level: number }[] {
	const out: { id: string; label: string; level: number }[] = [];
	for (const b of blocks) {
		if (b.type !== "heading") {
			continue;
		}
		const label = b.text.trim() || "(empty heading)";
		out.push({ id: b.id, label, level: b.level });
	}
	return out;
}

export function OutlinePanel() {
	const { blocks, scrollToBlock } = usePaperEditor();
	const entries = outlineEntries(blocks);

	return (
		<Flex.Column
			className="min-h-0 border-border border-r bg-muted/20"
			fullHeight
			gap={3}
		>
			<Flex.Row align="center" className="shrink-0" gap={2} padding={3}>
				<ListTreeIcon aria-hidden className="size-4 text-muted-foreground" />
				<Typography.H3 variant="sectionHeading">Outline</Typography.H3>
			</Flex.Row>
			<ScrollArea
				className="min-h-0 flex-1 px-2 pb-3"
				scrollFade
				scrollbarGutter
			>
				{entries.length === 0 ? (
					<Typography.Small className="px-2" variant="muted">
						Add a heading block to see the outline here.
					</Typography.Small>
				) : (
					<nav aria-label="Document outline">
						<ul className="m-0 flex list-none flex-col gap-0.5 p-0">
							{entries.map((e) => (
								<li key={e.id}>
									<button
										className="w-full rounded-md px-2 py-1.5 text-left text-foreground text-xs leading-snug hover:bg-accent hover:text-accent-foreground"
										style={{ paddingLeft: `${(e.level - 1) * 8 + 8}px` }}
										type="button"
										onClick={() => scrollToBlock(e.id)}
									>
										<Typography.Span
											className="line-clamp-2"
											variant="foreground"
										>
											{e.label}
										</Typography.Span>
									</button>
								</li>
							))}
						</ul>
					</nav>
				)}
			</ScrollArea>
		</Flex.Column>
	);
}
