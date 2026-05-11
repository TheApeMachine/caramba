"use client";

import { useStore } from "@tanstack/react-form";
import { PaperEditorProvider, usePaperEditor } from "#/components/latex/context";
import type { PaperMetadata } from "#/components/latex/model/types";
import { WritingCanvas } from "#/components/latex/panels/writing-canvas";

import { Flex } from "#/components/ui/flex";
import { LatexToolbar } from "./toolbar";

/*
PaperContextSnapshot renders a hidden element tagged with data-context so the
assistant's page context scraper picks up the current paper state on every send.
It stays invisible — sr-only keeps it accessible but out of view.
*/
function PaperContextSnapshot() {
	const { blocks, metadataForm } = usePaperEditor();

	const meta = useStore(metadataForm.store, (s) => s.values as PaperMetadata);

	const summary = [
		meta.title && `Title: ${meta.title}`,
		meta.authors && `Authors: ${meta.authors.replace(/\n/g, ", ")}`,
		`Blocks (${blocks.length}):`,
		...blocks.map((b, i) => {
			if (b.type === "heading") return `  [${i}] heading H${b.level} id=${b.id}: ${b.text}`;
			if (b.type === "paragraph") return `  [${i}] paragraph id=${b.id}: ${b.text.slice(0, 80)}${b.text.length > 80 ? "…" : ""}`;
			return `  [${i}] equation id=${b.id}: ${b.latex.slice(0, 60)}${b.latex.length > 60 ? "…" : ""}`;
		}),
	].filter(Boolean).join("\n");

	return (
		<span
			className="sr-only"
			data-context="Research paper content"
			data-context-key="paper_content"
			data-context-type="text"
			aria-hidden="true"
		>
			{summary}
		</span>
	);
}

export function PaperEditorApp() {
	return (
		<PaperEditorProvider>
			<PaperContextSnapshot />
			<Flex.Column className="box-border min-h-0 bg-background" fullHeight>
				<LatexToolbar />
				<Flex.Row className="min-h-0 flex-1 overflow-auto lg:flex-row lg:overflow-hidden">
					<Flex.Column
						className="max-h-44 min-h-0 w-full shrink-0 border-border border-b lg:h-full lg:max-h-none lg:w-52 lg:border-r lg:border-b-0"
						fullHeight
					>
					</Flex.Column>
					<Flex.Column
						className="min-h-[min(55dvh,560px)] flex-1 lg:min-h-0"
						fullHeight
					>
						<WritingCanvas />
					</Flex.Column>
					<Flex.Column
						className="min-h-[min(50dvh,420px)] w-full shrink-0 border-border border-t lg:h-full lg:min-h-0 lg:w-80 lg:border-l lg:border-t-0"
						fullHeight
					>
					</Flex.Column>
				</Flex.Row>
			</Flex.Column>
		</PaperEditorProvider>
	);
}
