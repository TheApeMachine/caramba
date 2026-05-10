"use client";

import { Link } from "@tanstack/react-router";
import { PaperEditorProvider } from "#/components/latex/context";
import { usePaperMetadataForm } from "#/components/latex/panels/metadata-tab";
import { OutlinePanel } from "#/components/latex/panels/outline-panel";
import { PaperEditorToolbar } from "#/components/latex/panels/paper-editor-toolbar";
import { RightPanel } from "#/components/latex/panels/right-panel";
import { WritingCanvas } from "#/components/latex/panels/writing-canvas";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

export function PaperEditorApp() {
	const metadataForm = usePaperMetadataForm();

	return (
		<PaperEditorProvider>
			<Flex.Column
				className="box-border min-h-0 flex-1 bg-background"
				fullHeight
			>
				<Flex.Row
					align="center"
					className="shrink-0 border-border border-b"
					justify="between"
					padding={4}
					gap={2}
				>
					<Flex.Column gap={1}>
						<Typography.PageTitle>Paper editor</Typography.PageTitle>
						<Typography.Paragraph variant="lead">
							Write in structure — export clean LaTeX when you are ready.
						</Typography.Paragraph>
					</Flex.Column>
					<Button render={<Link to="/research" />} size="sm" variant="outline">
						Back to research
					</Button>
				</Flex.Row>
				<PaperEditorToolbar />
				<Flex.Column className="min-h-0 flex-1 overflow-auto lg:flex-row lg:overflow-hidden">
					<Flex.Column
						className="max-h-44 min-h-0 w-full shrink-0 border-border border-b lg:h-full lg:max-h-none lg:w-[220px] lg:shrink-0 lg:border-r lg:border-b-0"
						fullHeight
					>
						<OutlinePanel />
					</Flex.Column>
					<Flex.Column
						className="min-h-[min(55dvh,560px)] flex-1 lg:min-h-0"
						fullHeight
					>
						<WritingCanvas />
					</Flex.Column>
					<Flex.Column
						className="min-h-[min(50dvh,420px)] w-full shrink-0 border-border border-t lg:h-full lg:min-h-0 lg:w-[min(100%,320px)] lg:shrink-0 lg:border-l lg:border-t-0"
						fullHeight
					>
						<RightPanel form={metadataForm} />
					</Flex.Column>
				</Flex.Column>
			</Flex.Column>
		</PaperEditorProvider>
	);
}
