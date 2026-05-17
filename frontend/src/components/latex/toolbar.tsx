"use client";

import { useStore } from "@tanstack/react-form";
import { InfoIcon, PlusIcon, TableOfContentsIcon } from "lucide-react";
import { usePaperEditor } from "#/components/latex/context";
import type { BlockKindDescriptor } from "#/components/latex/model/block-catalog";
import type { PaperMetadata } from "#/components/latex/model/types";
import { BlockKindMenu } from "#/components/latex/panels/block-kind-menu";
import { MetadataTab } from "#/components/latex/panels/metadata-tab";
import { OutlinePanel } from "#/components/latex/panels/outline-panel";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";
import {
	Sheet,
	SheetClose,
	SheetHeader,
	SheetPanel,
	SheetPopup,
	SheetTitle,
	SheetTrigger,
} from "#/components/ui/sheet";
import {
	Toolbar,
	ToolbarButton,
	ToolbarGroup,
	ToolbarSeparator,
} from "#/components/ui/toolbar";
import {
	Tooltip,
	TooltipPopup,
	TooltipProvider,
	TooltipTrigger,
} from "#/components/ui/tooltip";
import { Typography } from "#/components/ui/typography";

function lastBlockId(
	blocks: ReturnType<typeof usePaperEditor>["blocks"],
): string | null {
	const focusable = blocks.at(-1);
	return focusable ? focusable.id : null;
}

function DocumentTitle() {
	const { metadataForm, blocks } = usePaperEditor();
	const title = useStore(
		metadataForm.store,
		(state) => (state.values as PaperMetadata).title,
	);

	const headingFallback = blocks.find(
		(block) => block.type === "heading" && block.level === 1,
	);

	const display =
		title?.trim() ||
		(headingFallback && "text" in headingFallback
			? headingFallback.text.trim()
			: "") ||
		"Untitled paper";

	return (
		<Typography.Small
			className="truncate px-2 text-foreground"
			variant="foreground"
		>
			{display}
		</Typography.Small>
	);
}

export const LatexToolbar = () => {
	const { blocks, focusedBlockId, insertBlockAfter, metadataForm } =
		usePaperEditor();

	const handleInsert = (descriptor: BlockKindDescriptor) => {
		const targetId = focusedBlockId ?? lastBlockId(blocks);

		if (!targetId) {
			return;
		}

		insertBlockAfter(targetId, descriptor.build());
	};

	return (
		<TooltipProvider>
			<Toolbar className="m-3">
				<ToolbarGroup>
					<Sheet>
						<Tooltip>
							<TooltipTrigger
								render={
									<ToolbarButton
										aria-label="Outline"
										render={
											<SheetTrigger
												render={<Button size="icon" variant="ghost" />}
											>
												<TableOfContentsIcon />
											</SheetTrigger>
										}
									/>
								}
							/>
							<TooltipPopup sideOffset={8}>Outline</TooltipPopup>
						</Tooltip>

						<SheetPopup side="left" variant="inset">
							<SheetHeader>
								<SheetTitle>Outline</SheetTitle>
							</SheetHeader>

							<SheetPanel className="p-0">
								<OutlinePanel />
							</SheetPanel>
						</SheetPopup>
					</Sheet>
				</ToolbarGroup>

				<ToolbarSeparator />

				<ToolbarGroup>
					<BlockKindMenu
						variant="trigger"
						onSelect={handleInsert}
						trigger={
							<ToolbarButton
								render={
									<Button size="sm" variant="outline">
										<PlusIcon />
										Insert block
									</Button>
								}
							/>
						}
					/>
				</ToolbarGroup>

				<Flex.Row align="center" className="min-w-0 flex-1" justify="center">
					<DocumentTitle />
				</Flex.Row>

				<ToolbarGroup>
					<Sheet>
						<Tooltip>
							<TooltipTrigger
								render={
									<ToolbarButton
										aria-label="Paper details"
										render={
											<SheetTrigger
												render={<Button size="icon" variant="ghost" />}
											>
												<InfoIcon />
											</SheetTrigger>
										}
									/>
								}
							/>
							<TooltipPopup sideOffset={8}>Paper details</TooltipPopup>
						</Tooltip>

						<SheetPopup variant="inset">
							<SheetHeader>
								<SheetTitle>Paper details</SheetTitle>
							</SheetHeader>

							<SheetPanel>
								<MetadataTab form={metadataForm} />
							</SheetPanel>

							<Flex.Row justify="end" padding={3}>
								<SheetClose render={<Button variant="ghost" />}>
									Close
								</SheetClose>
							</Flex.Row>
						</SheetPopup>
					</Sheet>
				</ToolbarGroup>
			</Toolbar>
		</TooltipProvider>
	);
};
