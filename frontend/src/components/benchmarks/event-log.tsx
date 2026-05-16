"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import type { RunEvent } from "./mock-stream";

const levelColor: Record<RunEvent["level"], string> = {
	info: "text-foreground/80",
	warn: "text-warning-foreground",
	error: "text-destructive-foreground",
};

const levelDot: Record<RunEvent["level"], string> = {
	info: "bg-primary/70",
	warn: "bg-warning",
	error: "bg-destructive",
};

const formatTime = (timestamp: number) =>
	new Date(timestamp).toLocaleTimeString(undefined, {
		hour: "2-digit",
		hour12: false,
		minute: "2-digit",
		second: "2-digit",
	});

/*
EventLog autoscrolls as new entries arrive so the user always sees the
latest output. Render-only — the parent owns the event list.
*/
export const EventLog = ({ events }: { events: RunEvent[] }) => {
	const scrollerRef = useRef<HTMLDivElement | null>(null);

	const eventsCount = events.length;
	useEffect(() => {
		const node = scrollerRef.current;
		if (!node || eventsCount === 0) return;
		node.scrollTop = node.scrollHeight;
	}, [eventsCount]);

	return (
		<div
			ref={scrollerRef}
			className="h-full overflow-y-auto rounded-2xl border bg-muted/20 p-3 font-mono text-xs"
		>
			{events.length === 0 ? (
				<div className="flex h-full items-center justify-center text-muted-foreground">
					No events yet.
				</div>
			) : null}
			<ul className="flex flex-col gap-1.5">
				{events.map((event) => (
					<li
						key={`${event.timestamp}-${event.message}`}
						className="flex items-start gap-2"
					>
						<span
							aria-hidden
							className={cn(
								"mt-1 size-1.5 shrink-0 rounded-full",
								levelDot[event.level],
							)}
						/>
						<span className="text-muted-foreground">
							{formatTime(event.timestamp)}
						</span>
						<span className={cn("flex-1", levelColor[event.level])}>
							{event.message}
						</span>
					</li>
				))}
			</ul>
		</div>
	);
};
