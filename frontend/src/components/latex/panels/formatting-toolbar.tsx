"use client";

import { BoldIcon, CodeIcon, ItalicIcon, SigmaIcon } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "#/components/ui/button";
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

type WrapShortcut = {
	id: string;
	label: string;
	prefix: string;
	suffix: string;
	icon: ReactNode;
};

const textShortcuts: WrapShortcut[] = [
	{ id: "bold", label: "Bold", prefix: "**", suffix: "**", icon: <BoldIcon /> },
	{
		id: "italic",
		label: "Italic",
		prefix: "*",
		suffix: "*",
		icon: <ItalicIcon />,
	},
	{
		id: "code",
		label: "Inline code",
		prefix: "`",
		suffix: "`",
		icon: <CodeIcon />,
	},
];

const mathShortcut: WrapShortcut = {
	id: "math",
	label: "Inline math",
	prefix: "$",
	suffix: "$",
	icon: <SigmaIcon />,
};

function editableForRange(range: Range): HTMLElement | null {
	const container = range.commonAncestorContainer;
	const element =
		container instanceof Element ? container : container.parentElement;

	return element?.closest<HTMLElement>("[contenteditable]") ?? null;
}

function wrapSelection(prefix: string, suffix: string) {
	const selection = window.getSelection();

	if (!selection || selection.rangeCount === 0) {
		return;
	}

	const range = selection.getRangeAt(0);
	const text = selection.toString();
	const wrapped = `${prefix}${text}${suffix}`;
	const editable = editableForRange(range);

	range.deleteContents();

	const textNode = document.createTextNode(wrapped);
	range.insertNode(textNode);

	const cursor = document.createRange();
	if (text.length > 0) {
		cursor.setStartAfter(textNode);
	} else {
		cursor.setStart(textNode, prefix.length);
	}
	cursor.collapse(true);
	selection.removeAllRanges();
	selection.addRange(cursor);
	editable?.dispatchEvent(new Event("input", { bubbles: true }));
}

function FormattingButton({ shortcut }: { shortcut: WrapShortcut }) {
	return (
		<Tooltip>
			<TooltipTrigger
				render={
					<ToolbarButton
						aria-label={shortcut.label}
						render={
							<Button
								onMouseDown={(event) => {
									event.preventDefault();
									wrapSelection(shortcut.prefix, shortcut.suffix);
								}}
								size="icon"
								variant="ghost"
							/>
						}
					>
						{shortcut.icon}
					</ToolbarButton>
				}
			/>
			<TooltipPopup sideOffset={8}>{shortcut.label}</TooltipPopup>
		</Tooltip>
	);
}

export function FormattingToolbar() {
	return (
		<TooltipProvider>
			<Toolbar
				className="absolute -top-12 right-0 z-10 shadow-md"
				onMouseDown={(event) => event.preventDefault()}
			>
				<ToolbarGroup>
					{textShortcuts.map((shortcut) => (
						<FormattingButton key={shortcut.id} shortcut={shortcut} />
					))}
				</ToolbarGroup>

				<ToolbarSeparator />

				<ToolbarGroup>
					<FormattingButton shortcut={mathShortcut} />
				</ToolbarGroup>
			</Toolbar>
		</TooltipProvider>
	);
}
