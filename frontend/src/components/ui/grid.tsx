import { cva, type VariantProps } from "class-variance-authority";
import {
	type HTMLMotionProps,
	type Transition,
	motion,
} from "motion/react";
import { cn } from "@/lib/utils";

const springPanel: Transition = {
	type: "spring",
	stiffness: 380,
	damping: 32,
	mass: 0.7,
};
const easeFast: Transition = { duration: 0.18, ease: [0.22, 1, 0.36, 1] };

type AppearVariant =
	| "fade"
	| "fadeUp"
	| "fadeDown"
	| "scaleIn";

const appearPresets: Record<
	AppearVariant,
	Pick<HTMLMotionProps<"div">, "initial" | "animate" | "exit" | "transition">
> = {
	fade: {
		initial: { opacity: 0 },
		animate: { opacity: 1 },
		exit: { opacity: 0 },
		transition: easeFast,
	},
	fadeUp: {
		initial: { opacity: 0, y: 8 },
		animate: { opacity: 1, y: 0 },
		exit: { opacity: 0, y: -8 },
		transition: easeFast,
	},
	fadeDown: {
		initial: { opacity: 0, y: -8 },
		animate: { opacity: 1, y: 0 },
		exit: { opacity: 0, y: 8 },
		transition: easeFast,
	},
	scaleIn: {
		initial: { opacity: 0, scale: 0.9 },
		animate: { opacity: 1, scale: 1 },
		exit: { opacity: 0, scale: 0.9 },
		transition: springPanel,
	},
};

export const gridVariants = cva("grid", {
	variants: {
		cols: {
			1: "grid-cols-1",
			2: "grid-cols-2",
			3: "grid-cols-3",
			4: "grid-cols-4",
			5: "grid-cols-5",
			6: "grid-cols-6",
			7: "grid-cols-7",
			8: "grid-cols-8",
			9: "grid-cols-9",
			10: "grid-cols-10",
			11: "grid-cols-11",
			12: "grid-cols-12",
			none: "grid-cols-none",
		},
		rows: {
			1: "grid-rows-1",
			2: "grid-rows-2",
			3: "grid-rows-3",
			4: "grid-rows-4",
			5: "grid-rows-5",
			6: "grid-rows-6",
			none: "grid-rows-none",
		},
		flow: {
			row: "grid-flow-row",
			col: "grid-flow-col",
			dense: "grid-flow-dense",
			rowDense: "grid-flow-row-dense",
			colDense: "grid-flow-col-dense",
		},
		justify: {
			start: "justify-items-start",
			center: "justify-items-center",
			end: "justify-items-end",
			stretch: "justify-items-stretch",
		},
		align: {
			start: "items-start",
			center: "items-center",
			end: "items-end",
			stretch: "items-stretch",
			baseline: "items-baseline",
		},
		justifyContent: {
			start: "justify-start",
			center: "justify-center",
			end: "justify-end",
			between: "justify-between",
			around: "justify-around",
			evenly: "justify-evenly",
		},
		alignContent: {
			start: "content-start",
			center: "content-center",
			end: "content-end",
			between: "content-between",
			around: "content-around",
			evenly: "content-evenly",
			stretch: "content-stretch",
		},
		gap: {
			1: "gap-1",
			2: "gap-2",
			3: "gap-3",
			4: "gap-4",
			5: "gap-5",
			6: "gap-6",
			7: "gap-7",
			8: "gap-8",
			9: "gap-9",
			10: "gap-10",
			11: "gap-11",
			12: "gap-12",
		},
		gapX: {
			1: "gap-x-1",
			2: "gap-x-2",
			3: "gap-x-3",
			4: "gap-x-4",
			5: "gap-x-5",
			6: "gap-x-6",
			7: "gap-x-7",
			8: "gap-x-8",
			9: "gap-x-9",
			10: "gap-x-10",
			11: "gap-x-11",
			12: "gap-x-12",
		},
		gapY: {
			1: "gap-y-1",
			2: "gap-y-2",
			3: "gap-y-3",
			4: "gap-y-4",
			5: "gap-y-5",
			6: "gap-y-6",
			7: "gap-y-7",
			8: "gap-y-8",
			9: "gap-y-9",
			10: "gap-y-10",
			11: "gap-y-11",
			12: "gap-y-12",
		},
		padding: {
			1: "p-1",
			2: "p-2",
			3: "p-3",
			4: "p-4",
			5: "p-5",
			6: "p-6",
			7: "p-7",
			8: "p-8",
			9: "p-9",
			10: "p-10",
			11: "p-11",
			12: "p-12",
		},
		margin: {
			1: "m-1",
			2: "m-2",
			3: "m-3",
			4: "m-4",
			5: "m-5",
			6: "m-6",
			7: "m-7",
			8: "m-8",
			9: "m-9",
			10: "m-10",
			11: "m-11",
			12: "m-12",
		},
		grow: {
			grow: "grow",
			shrink: "shrink",
			growShrink: "grow shrink",
		},
		fullHeight: {
			fullHeight: "h-full",
		},
		fullWidth: {
			fullWidth: "w-full",
		},
	},
});

type GridVariantProps = VariantProps<typeof gridVariants>;

type ColCount = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
type RowCount = 1 | 2 | 3 | 4 | 5 | 6;
type Spacing = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;

export const Grid = ({
	children,
	className,
	cols,
	rows,
	flow,
	justify,
	align,
	justifyContent,
	alignContent,
	gap,
	gapX,
	gapY,
	padding,
	margin,
	grow,
	fullHeight,
	fullWidth,
	appear,
	...props
}: HTMLMotionProps<"div"> &
	Omit<GridVariantProps, "fullHeight" | "fullWidth"> & {
		cols?: ColCount | "none";
		rows?: RowCount | "none";
		flow?: "row" | "col" | "dense" | "rowDense" | "colDense";
		justify?: "start" | "center" | "end" | "stretch";
		align?: "start" | "center" | "end" | "stretch" | "baseline";
		justifyContent?:
			| "start"
			| "center"
			| "end"
			| "between"
			| "around"
			| "evenly";
		alignContent?:
			| "start"
			| "center"
			| "end"
			| "between"
			| "around"
			| "evenly"
			| "stretch";
		gap?: Spacing;
		gapX?: Spacing;
		gapY?: Spacing;
		padding?: Spacing;
		margin?: Spacing;
		grow?: "grow" | "shrink" | "growShrink";
		fullHeight?: boolean;
		fullWidth?: boolean;
		appear?: AppearVariant;
	}) => {
	const preset = appear ? appearPresets[appear] : undefined;

	return (
		<motion.div
			className={cn(
				gridVariants({
					cols,
					rows,
					flow,
					justify,
					align,
					justifyContent,
					alignContent,
					gap,
					gapX,
					gapY,
					padding,
					margin,
					grow,
					fullHeight: fullHeight ? "fullHeight" : undefined,
					fullWidth: fullWidth ? "fullWidth" : undefined,
				}),
				className,
			)}
			{...preset}
			{...props}
		>
			{children}
		</motion.div>
	);
};

export const gridItemVariants = cva("", {
	variants: {
		colSpan: {
			1: "col-span-1",
			2: "col-span-2",
			3: "col-span-3",
			4: "col-span-4",
			5: "col-span-5",
			6: "col-span-6",
			7: "col-span-7",
			8: "col-span-8",
			9: "col-span-9",
			10: "col-span-10",
			11: "col-span-11",
			12: "col-span-12",
			full: "col-span-full",
		},
		rowSpan: {
			1: "row-span-1",
			2: "row-span-2",
			3: "row-span-3",
			4: "row-span-4",
			5: "row-span-5",
			6: "row-span-6",
			full: "row-span-full",
		},
		colStart: {
			1: "col-start-1",
			2: "col-start-2",
			3: "col-start-3",
			4: "col-start-4",
			5: "col-start-5",
			6: "col-start-6",
			7: "col-start-7",
			8: "col-start-8",
			9: "col-start-9",
			10: "col-start-10",
			11: "col-start-11",
			12: "col-start-12",
			13: "col-start-13",
			auto: "col-start-auto",
		},
		colEnd: {
			1: "col-end-1",
			2: "col-end-2",
			3: "col-end-3",
			4: "col-end-4",
			5: "col-end-5",
			6: "col-end-6",
			7: "col-end-7",
			8: "col-end-8",
			9: "col-end-9",
			10: "col-end-10",
			11: "col-end-11",
			12: "col-end-12",
			13: "col-end-13",
			auto: "col-end-auto",
		},
		rowStart: {
			1: "row-start-1",
			2: "row-start-2",
			3: "row-start-3",
			4: "row-start-4",
			5: "row-start-5",
			6: "row-start-6",
			7: "row-start-7",
			auto: "row-start-auto",
		},
		rowEnd: {
			1: "row-end-1",
			2: "row-end-2",
			3: "row-end-3",
			4: "row-end-4",
			5: "row-end-5",
			6: "row-end-6",
			7: "row-end-7",
			auto: "row-end-auto",
		},
		justifySelf: {
			start: "justify-self-start",
			center: "justify-self-center",
			end: "justify-self-end",
			stretch: "justify-self-stretch",
			auto: "justify-self-auto",
		},
		alignSelf: {
			start: "self-start",
			center: "self-center",
			end: "self-end",
			stretch: "self-stretch",
			baseline: "self-baseline",
			auto: "self-auto",
		},
		padding: {
			1: "p-1",
			2: "p-2",
			3: "p-3",
			4: "p-4",
			5: "p-5",
			6: "p-6",
			7: "p-7",
			8: "p-8",
			9: "p-9",
			10: "p-10",
			11: "p-11",
			12: "p-12",
		},
	},
});

type GridItemVariantProps = VariantProps<typeof gridItemVariants>;

type ColSpan = ColCount | "full";
type RowSpan = RowCount | "full";
type ColLine = ColCount | 13 | "auto";
type RowLine = RowCount | 7 | "auto";

Grid.Item = ({
	children,
	className,
	colSpan,
	rowSpan,
	colStart,
	colEnd,
	rowStart,
	rowEnd,
	justifySelf,
	alignSelf,
	padding,
	appear,
	...props
}: HTMLMotionProps<"div"> &
	GridItemVariantProps & {
		colSpan?: ColSpan;
		rowSpan?: RowSpan;
		colStart?: ColLine;
		colEnd?: ColLine;
		rowStart?: RowLine;
		rowEnd?: RowLine;
		justifySelf?: "start" | "center" | "end" | "stretch" | "auto";
		alignSelf?: "start" | "center" | "end" | "stretch" | "baseline" | "auto";
		padding?: Spacing;
		appear?: AppearVariant;
	}) => {
	const preset = appear ? appearPresets[appear] : undefined;

	return (
		<motion.div
			className={cn(
				gridItemVariants({
					colSpan,
					rowSpan,
					colStart,
					colEnd,
					rowStart,
					rowEnd,
					justifySelf,
					alignSelf,
					padding,
				}),
				className,
			)}
			{...preset}
			{...props}
		>
			{children}
		</motion.div>
	);
};
