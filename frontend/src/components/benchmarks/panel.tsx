import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface PanelProps {
	title: string;
	hint?: string;
	action?: ReactNode;
	children: ReactNode;
	className?: string;
	bodyClassName?: string;
}

/*
Panel is the live-run chart card. Same chrome as the dashboard widgets but
without drag/resize affordances — runs are read-only views, not user-edited
layouts. Header + body, body sized via flex-1 + min-h-0 so charts can fill.
*/
export const Panel = ({
	title,
	hint,
	action,
	children,
	className,
	bodyClassName,
}: PanelProps) => (
	<div
		className={cn(
			"flex h-full min-h-0 flex-col overflow-hidden rounded-2xl border bg-card/40",
			className,
		)}
	>
		<header className="flex items-start justify-between gap-3 border-b bg-muted/20 px-4 py-2.5">
			<div className="flex min-w-0 flex-col">
				<span className="font-medium text-foreground text-sm">{title}</span>
				{hint ? (
					<span className="text-muted-foreground text-xs">{hint}</span>
				) : null}
			</div>
			{action}
		</header>
		<div className={cn("relative flex min-h-0 flex-1 p-3", bodyClassName)}>
			{children}
		</div>
	</div>
);
