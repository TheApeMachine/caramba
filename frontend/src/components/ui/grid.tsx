import type React from "react";
import { cn } from "@/lib/utils";
import { Flex } from "./flex";

export const Grid = ({
	children,
	className,
}: {
	children: React.ReactNode;
	className?: string;
}) => {
	return (
		<div
			className={cn(
				"grid flex-1 items-stretch gap-9 pb-12 lg:grid-cols-2 lg:gap-6 xl:gap-9",
				className,
			)}
		>
			{children}
		</div>
	);
};

Grid.Item = ({ children }: { children: React.ReactNode }) => {
	return (
		<Flex.Row className="relative min-w-0 **:data-[slot=preview]:w-full **:data-[slot=preview]:flex **:data-[slot=preview]:justify-center">
			<Flex.Column
				className="relative rounded-2xl border bg-card not-dark:bg-clip-padding text-card-foreground shadow-xs/5 [--clip-bottom:-1rem] [--clip-top:-1rem] before:pointer-events-none before:absolute before:inset-0 before:rounded-[calc(var(--radius-2xl)-1px)] before:bg-muted/72 before:shadow-[0_1px_--theme(--color-black/4%)] has-data-[slot=table-container]:overflow-hidden *:data-[slot=card]:-m-px *:data-[slot=table-container]:-m-px *:data-[slot=table-container]:w-[calc(100%+2px)] *:not-first:data-[slot=card]:rounded-t-xl *:not-last:data-[slot=card]:rounded-b-xl *:data-[slot=card]:bg-clip-padding *:data-[slot=card]:shadow-none *:data-[slot=card]:before:hidden *:not-first:data-[slot=card]:before:rounded-t-[calc(var(--radius-xl)-1px)] *:not-last:data-[slot=card]:before:rounded-b-[calc(var(--radius-xl)-1px)] dark:before:shadow-[0_-1px_--theme(--color-white/6%)] *:data-[slot=card]:[clip-path:inset(var(--clip-top)_1px_var(--clip-bottom)_1px_round_calc(var(--radius-2xl)-1px))] *:data-[slot=card]:last:[--clip-bottom:1px] *:data-[slot=card]:first:[--clip-top:1px] w-full after:pointer-events-none after:absolute after:inset-[-5px] after:-z-1 after:rounded-[calc(var(--radius-xl)+4px)] after:border after:border-border/64 dark:bg-background"
				data-slot="card-frame"
			>
				<Flex.Row
					className="relative rounded-2xl border bg-card not-dark:bg-clip-padding text-card-foreground shadow-xs/5 before:pointer-events-none before:absolute before:inset-0 before:rounded-[calc(var(--radius-2xl)-1px)] before:shadow-[0_1px_--theme(--color-black/4%)] dark:before:shadow-[0_-1px_--theme(--color-white/6%)] min-h-50 flex-1 flex-col flex-wrap overflow-x-auto dark:bg-background"
					data-slot="card"
				>
					{children}
				</Flex.Row>
			</Flex.Column>
		</Flex.Row>
	);
};
