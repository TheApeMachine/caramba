"use client";

import { useStore } from "@tanstack/react-form";
import {
	PaperEditorProvider,
	usePaperEditor,
} from "#/components/latex/context";
import type { PaperMetadata } from "#/components/latex/model/types";
import { WritingCanvas } from "#/components/latex/panels/writing-canvas";

import { DragDropProvider } from "#/components/ui/drag-drop";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { LatexToolbar } from "./toolbar";

export type PaperEditorAppProps = {
	paperId?: string;
	projectId?: string;
	onPaperBootstrapped?: (paperId: string) => void;
};

const PaperContextSnapshot = () => {
	const { blocks, metadataForm } = usePaperEditor();

	const meta = useStore(metadataForm.store, (s) => s.values as PaperMetadata);

	const summary = [
		meta.title && `Title: ${meta.title}`,
		meta.authors && `Authors: ${meta.authors.replace(/\n/g, ", ")}`,
		`Blocks (${blocks.length}):`,
		...blocks.map((block, index) => {
			if (block.type === "heading") {
				return `  [${index}] heading H${block.level} id=${block.id}: ${block.text}`;
			}

			if (block.type === "paragraph") {
				const preview = block.text.slice(0, 80);
				const ellipsis = block.text.length > 80 ? "…" : "";
				return `  [${index}] paragraph id=${block.id}: ${preview}${ellipsis}`;
			}

			if (block.type === "list") {
				const lines = block.text.split("\n");
				const markedItems = lines.filter((line) =>
					/^\s*(?:\d+\.\s+|[-*+]\s+)/.test(line),
				).length;
				const items = markedItems || lines.filter(Boolean).length;
				const kind = block.ordered ? "ordered" : "unordered";
				return `  [${index}] list (${kind}, ${items} items) id=${block.id}`;
			}

			const eqPreview = block.latex.slice(0, 60);
			const eqEllipsis = block.latex.length > 60 ? "…" : "";
			return `  [${index}] equation id=${block.id}: ${eqPreview}${eqEllipsis}`;
		}),
	]
		.filter(Boolean)
		.join("\n");

	return (
		<span
			aria-hidden="true"
			className="sr-only"
			data-context="Research paper content"
			data-context-key="paper_content"
			data-context-type="text"
		>
			{summary}
		</span>
	);
};

const PaperEditorShell = () => {
	const { paperPersistence } = usePaperEditor();

	if (paperPersistence.enabled && paperPersistence.bootstrapError) {
		return (
			<Flex.Center className="min-h-[50dvh] flex-1 p-4">
				<Typography.Paragraph className="text-destructive text-center">
					{paperPersistence.bootstrapError}
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	if (paperPersistence.enabled && !paperPersistence.ready) {
		const message = paperPersistence.waitingForRemote
			? "Loading paper from server…"
			: "Preparing editor…";

		return (
			<Flex.Center className="min-h-[50dvh] flex-1 p-4">
				<Typography.Paragraph variant="muted">{message}</Typography.Paragraph>
			</Flex.Center>
		);
	}

	return (
		<>
			<PaperContextSnapshot />

			<DragDropProvider>
				<Flex.Column className="box-border min-h-0 bg-background" fullHeight>
					<LatexToolbar />

					<Flex.Column className="min-h-0 flex-1" fullHeight>
						<WritingCanvas />
					</Flex.Column>
				</Flex.Column>
			</DragDropProvider>
		</>
	);
};

export const PaperEditorApp = ({
	paperId,
	projectId,
	onPaperBootstrapped,
}: PaperEditorAppProps) => {
	return (
		<PaperEditorProvider
			bootstrapProjectId={projectId}
			onPaperBootstrapped={onPaperBootstrapped}
			paperId={paperId}
		>
			<PaperEditorShell />
		</PaperEditorProvider>
	);
};
