"use client";

import {
	type ClipboardEventHandler,
	type FocusEventHandler,
	type FormEventHandler,
	forwardRef,
	type KeyboardEventHandler,
	useEffect,
	useImperativeHandle,
	useRef,
} from "react";
import { cn } from "@/lib/utils";

export type EditableBlockProps = {
	value: string;
	onChange: (value: string) => void;
	onKeyDown?: KeyboardEventHandler<HTMLDivElement>;
	onFocus?: FocusEventHandler<HTMLDivElement>;
	onBlur?: FocusEventHandler<HTMLDivElement>;
	className?: string;
	placeholder?: string;
	ariaLabel?: string;
};

/*
EditableBlock is a contenteditable shell that behaves like a controlled
plain-text input without the chrome of a textarea. Outside-driven changes
to `value` are written into the element only when they differ from the
current DOM text, so typing never resets the caret.
*/
export const EditableBlock = forwardRef<HTMLDivElement, EditableBlockProps>(
	function EditableBlock(
		{
			value,
			onChange,
			onKeyDown,
			onFocus,
			onBlur,
			className,
			placeholder,
			ariaLabel,
		},
		ref,
	) {
		const innerRef = useRef<HTMLDivElement | null>(null);

		useImperativeHandle(ref, () => innerRef.current as HTMLDivElement, []);

		useEffect(() => {
			const el = innerRef.current;

			if (!el) {
				return;
			}

			if (el.innerText !== value) {
				el.innerText = value;
			}
		}, [value]);

		const handleInput: FormEventHandler<HTMLDivElement> = (event) => {
			onChange((event.currentTarget as HTMLDivElement).innerText);
		};

		const handlePaste: ClipboardEventHandler<HTMLDivElement> = (event) => {
			event.preventDefault();
			const text = event.clipboardData.getData("text/plain");
			const editable = event.currentTarget;
			const selection = window.getSelection();

			editable.focus();

			if (!selection) {
				return;
			}

			const range =
				selection.rangeCount > 0
					? selection.getRangeAt(0)
					: document.createRange();

			if (!editable.contains(range.commonAncestorContainer)) {
				range.selectNodeContents(editable);
				range.collapse(false);
			}

			range.deleteContents();

			const textNode = document.createTextNode(text);
			range.insertNode(textNode);
			range.setStartAfter(textNode);
			range.collapse(true);

			selection.removeAllRanges();
			selection.addRange(range);
			onChange(editable.innerText);
		};

		return (
			// biome-ignore lint/a11y/useSemanticElements: contenteditable div is the whole point of this primitive
			<div
				aria-label={ariaLabel}
				className={cn(
					"whitespace-pre-wrap wrap-break-word outline-none",
					"empty:before:pointer-events-none empty:before:text-muted-foreground empty:before:content-[attr(data-placeholder)]",
					className,
				)}
				contentEditable="plaintext-only"
				data-placeholder={placeholder}
				onBlur={onBlur}
				onFocus={onFocus}
				onInput={handleInput}
				onKeyDown={onKeyDown}
				onPaste={handlePaste}
				ref={innerRef}
				role="textbox"
				suppressContentEditableWarning
				tabIndex={0}
			/>
		);
	},
);

/*
Caret offset within an element, counted as plain-text characters.
Useful for splitting block content on Enter.
*/
export function caretOffset(el: HTMLElement): number {
	const selection = window.getSelection();

	if (!selection || selection.rangeCount === 0) {
		return 0;
	}

	const range = selection.getRangeAt(0);

	if (!el.contains(range.endContainer)) {
		return 0;
	}

	const measure = range.cloneRange();
	measure.selectNodeContents(el);
	measure.setEnd(range.endContainer, range.endOffset);

	return measure.toString().length;
}
