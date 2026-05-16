"use client";

import type { DragEvent } from "react";

interface ResizeHandleProps {
	widgetId: string;
	onDragStart: () => void;
}

/*
ResizeHandle is the corner grip on a placed widget. It uses the same
HTML5 drag system as widget moves: the cells (empty and placed) already
fire dragenter/dragover and accept drops, so the handle just signals
"this is a resize drag" via the parent's payload ref. The drop target's
(col, row) then defines the new span relative to the widget's anchor.
*/
export const ResizeHandle = ({ widgetId, onDragStart }: ResizeHandleProps) => {
	const handleDragStart = (event: DragEvent<HTMLButtonElement>) => {
		event.stopPropagation();
		event.dataTransfer.effectAllowed = "move";
		event.dataTransfer.setData("text/plain", `resize:${widgetId}`);
		onDragStart();
		event.currentTarget.classList.add("dash-dragging");
	};

	const handleDragEnd = (event: DragEvent<HTMLButtonElement>) => {
		event.currentTarget.classList.remove("dash-dragging");
	};

	return (
		<button
			type="button"
			draggable
			data-dash-resize
			aria-label="Resize widget"
			onDragStart={handleDragStart}
			onDragEnd={handleDragEnd}
			className="absolute bottom-1 right-1 z-10 h-4 w-4 cursor-nwse-resize rounded-sm border border-muted-foreground/40 bg-background/70 opacity-0 transition group-hover:opacity-100"
		/>
	);
};
