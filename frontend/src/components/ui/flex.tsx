import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const flexVariants = cva("flex", {
	variants: {
		direction: {
			row: "flex-row",
			column: "flex-col",
			"row-reverse": "flex-row-reverse",
			"column-reverse": "flex-col-reverse",
		},
		align: {
			start: "items-start",
			end: "items-end",
			center: "items-center",
			baseline: "items-baseline",
			stretch: "items-stretch",
		},
		justify: {
			start: "justify-start",
			end: "justify-end",
			center: "justify-center",
			between: "justify-between",
			around: "justify-around",
			evenly: "justify-evenly",
		},
		wrap: {
			nowrap: "flex-nowrap",
			wrap: "flex-wrap",
			"wrap-reverse": "flex-wrap-reverse",
		},
		gap: {
			0: "gap-0",
			2: "gap-2",
			4: "gap-4",
			6: "gap-6",
			8: "gap-8",
		},
		pad: {
			0: "p-0",
			2: "p-2",
			4: "p-4",
			6: "p-6",
			8: "p-8",
		},
		grow: {
			true: "flex-grow",
			false: "flex-grow-0",
		},
		shrink: {
			true: "flex-shrink",
			false: "flex-shrink-0",
		},
		fullWidth: {
			true: "w-full",
			false: "w-auto",
		},
		fullHeight: {
			true: "h-full",
			false: "h-auto",
		},
	},
	defaultVariants: {
		direction: "row",
	},
});

interface FlexProps
	extends React.HTMLAttributes<HTMLDivElement>,
		VariantProps<typeof flexVariants> {
	ref?: React.Ref<HTMLDivElement>;
}

export const Flex = ({
	direction = "row",
	align,
	justify,
	wrap,
	gap,
	pad,
	grow,
	shrink,
	fullWidth,
	fullHeight,
	className,
	children,
	ref,
	...props
}: FlexProps) => {
	return (
		<div
			ref={ref}
			className={cn(
				className,
				flexVariants({
					direction,
					align,
					justify,
					wrap,
					gap,
					pad,
					grow,
					shrink,
					fullWidth,
					fullHeight,
				}),
			)}
			{...props}
		>
			{children}
		</div>
	);
};
