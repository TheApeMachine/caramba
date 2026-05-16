"use client";

import { useDragDrop } from "@/components/ui/drag-drop";

interface ResizeHandleProps {
	widgetId: string;
}

/*
ResizeHandle is the corner grip on a placed widget. It begins a drag with
a `resize` payload; the existing drop targets (cells) read that payload
and treat the drop coordinate as the new bottom-right corner of the
widget's span. Uses the shared drag-drop context, not a separate channel.
*/
export const ResizeHandle = ({ widgetId }: ResizeHandleProps) => {
	const dnd = useDragDrop<{ source: "resize"; widgetId: string }>();

	return (
		<button
			type="button"
			draggable
			data-dash-resize
			aria-label="Resize widget"
			onDragStart={(event) => {
				event.stopPropagation();
				event.dataTransfer.effectAllowed = "move";
				event.dataTransfer.setData("text/plain", "");
				dnd.begin({ source: "resize", widgetId });
				event.currentTarget.classList.add("dash-dragging");
			}}
			onDragEnd={(event) => {
				event.currentTarget.classList.remove("dash-dragging");
			}}
			className="absolute bottom-1 right-1 z-10 h-4 w-4 cursor-nwse-resize rounded-sm border border-muted-foreground/40 bg-background/70 shadow-sm transition hover:bg-background"
		/>
	);
};
