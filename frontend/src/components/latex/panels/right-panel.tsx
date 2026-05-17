"use client";

import { PanelRight } from "lucide-react";
import { usePaperEditor } from "#/components/latex/context";
import { ExportTab } from "#/components/latex/panels/export-tab";
import { MetadataTab } from "#/components/latex/panels/metadata-tab";
import { Flex } from "#/components/ui/flex";
import { Tabs } from "#/components/ui/tabs";
import { Typography } from "#/components/ui/typography";

export function RightPanel() {
	const { metadataForm } = usePaperEditor();

	return (
		<Flex.Column className="min-h-0 bg-muted/15" fullHeight>
			<Flex.Row
				align="center"
				className="shrink-0 border-border border-b"
				gap={2}
				padding={3}
			>
				<PanelRight aria-hidden className="size-4 text-muted-foreground" />
				<Typography.H3 variant="sectionHeading">Paper details</Typography.H3>
			</Flex.Row>

			<Tabs className="min-h-0 flex-1 gap-0" defaultValue="metadata">
				<Tabs.List
					className="w-full shrink-0 justify-start rounded-none border-border border-b bg-transparent px-2"
					variant="underline"
				>
					<Tabs.Tab value="metadata">Metadata</Tabs.Tab>
					<Tabs.Tab value="export">Export</Tabs.Tab>
				</Tabs.List>

				<Tabs.Panel
					className="min-h-0 flex-1 data-[orientation=horizontal]:pb-0"
					value="metadata"
				>
					<MetadataTab form={metadataForm} />
				</Tabs.Panel>

				<Tabs.Panel
					className="flex min-h-0 flex-1 flex-col data-[orientation=horizontal]:pb-0"
					value="export"
				>
					<ExportTab form={metadataForm} />
				</Tabs.Panel>
			</Tabs>
		</Flex.Column>
	);
}
