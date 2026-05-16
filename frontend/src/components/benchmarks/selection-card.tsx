import { CheckIcon } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface SelectionCardProps {
	selected: boolean;
	onSelect: () => void;
	title: string;
	subtitle?: string;
	hint?: string;
	icon?: ReactNode;
	disabled?: boolean;
}

/*
SelectionCard is the wizard's primary control. Large, low-friction targets
beat a forest of radio buttons when the researcher is making a categorical
choice (model / dataset / backend). Selected state earns a strong border, a
checkmark, and a subtle background lift so it reads at a glance.
*/
export const SelectionCard = ({
	selected,
	onSelect,
	title,
	subtitle,
	hint,
	icon,
	disabled,
}: SelectionCardProps) => (
	<button
		type="button"
		onClick={onSelect}
		disabled={disabled}
		aria-pressed={selected}
		className={cn(
			"group relative flex h-full w-full flex-col gap-1 rounded-xl border bg-card/40 p-3 text-left transition",
			"hover:border-primary/60 hover:bg-card/80",
			"focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
			selected && "border-primary bg-primary/5 shadow-sm",
			disabled && "pointer-events-none opacity-50",
		)}
	>
		<div className="flex items-start justify-between gap-2">
			<div className="flex min-w-0 items-center gap-2">
				{icon ? (
					<span className="text-muted-foreground group-hover:text-foreground">
						{icon}
					</span>
				) : null}
				<span className="truncate font-medium text-sm text-foreground">
					{title}
				</span>
			</div>
			<span
				className={cn(
					"flex size-5 shrink-0 items-center justify-center rounded-full border transition",
					selected
						? "border-primary bg-primary text-primary-foreground"
						: "border-muted-foreground/30 bg-transparent text-transparent",
				)}
				aria-hidden
			>
				<CheckIcon className="size-3" />
			</span>
		</div>
		{subtitle ? (
			<span className="text-muted-foreground text-xs">{subtitle}</span>
		) : null}
		{hint ? (
			<span className="text-muted-foreground/80 text-xs leading-snug">
				{hint}
			</span>
		) : null}
	</button>
);
