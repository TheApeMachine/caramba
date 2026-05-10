"use client";

import { useStore } from "@tanstack/react-form";
import { useCallback, useMemo, useState } from "react";
import { usePaperEditor } from "#/components/latex/context";
import { exportPaperToLatex } from "#/components/latex/model/latex-export";
import type { PaperMetadata } from "#/components/latex/model/types";
import type { PaperMetadataFormApi } from "#/components/latex/panels/metadata-tab";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";
import { ScrollArea } from "#/components/ui/scroll-area";
import { Typography } from "#/components/ui/typography";

export const ExportTab = ({ form }: { form: PaperMetadataFormApi }) => {
	const { blocks } = usePaperEditor();
	const [copied, setCopied] = useState(false);

	const meta = useStore(form.store, (s) => s.values as PaperMetadata);
	const latex = useMemo(() => exportPaperToLatex(meta, blocks), [meta, blocks]);

	const copy = useCallback(async () => {
		try {
			await navigator.clipboard.writeText(latex);
			setCopied(true);
			globalThis.setTimeout(() => setCopied(false), 2000);
		} catch {
			setCopied(false);
		}
	}, [latex]);

	return (
		<Flex.Column gap={3} padding={3} className="min-h-0" fullHeight>
			<Flex.Row align="center" justify="between" gap={2} wrap="wrap">
				<Typography.Small className="max-w-prose" variant="muted">
					LaTeX for your Go toolchain — not compiled in the browser.
				</Typography.Small>
				<Button
					size="sm"
					type="button"
					variant="outline"
					onClick={() => {
						void copy();
					}}
				>
					{copied ? "Copied" : "Copy LaTeX"}
				</Button>
			</Flex.Row>
			<ScrollArea
				className="min-h-0 flex-1 rounded-lg border border-border bg-muted/30"
				scrollFade
				scrollbarGutter
			>
				<Typography.Pre className="p-3" variant="codeExport">
					{latex}
				</Typography.Pre>
			</ScrollArea>
		</Flex.Column>
	);
};
