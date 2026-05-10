"use client";

import { PanelRight } from "lucide-react";
import { ExportTab } from "#/components/latex/panels/export-tab";
import {
	MetadataTab,
	type PaperMetadataFormApi,
} from "#/components/latex/panels/metadata-tab";
import { Flex } from "#/components/ui/flex";
import { Tabs, TabsList, TabsPanel, TabsTab } from "#/components/ui/tabs";
import { Typography } from "#/components/ui/typography";

export function RightPanel({ form }: { form: PaperMetadataFormApi }) {
	return (
		<Flex.Column
			className="min-h-0 border-border border-l bg-muted/15"
			fullHeight
		>
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
				<TabsList
					className="w-full shrink-0 justify-start rounded-none border-border border-b bg-transparent px-2"
					variant="underline"
				>
					<TabsTab value="metadata">Metadata</TabsTab>
					<TabsTab value="export">Export</TabsTab>
				</TabsList>
				<TabsPanel
					className="min-h-0 flex-1 data-[orientation=horizontal]:pb-0"
					value="metadata"
				>
					<MetadataTab form={form} />
				</TabsPanel>
				<TabsPanel
					className="flex min-h-0 flex-1 flex-col data-[orientation=horizontal]:pb-0"
					value="export"
				>
					<ExportTab form={form} />
				</TabsPanel>
			</Tabs>
		</Flex.Column>
	);
}
