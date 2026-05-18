import type {
	AstNode,
	MathAstBuilder,
} from "#/components/latex/math-subselection/ast";

export type StructureTransform = {
	id: string;
	icon: string;
	label: string;
	apply: (builder: MathAstBuilder, expr: AstNode) => AstNode;
};

/*
STRUCTURE_TRANSFORMS mirrors docs/math_subselection_developed_demo.html txns.
*/
export const STRUCTURE_TRANSFORMS: StructureTransform[] = [
	{
		id: "square",
		icon: "x²",
		label: "Square it",
		apply: (builder, expr) => builder.pow(expr, builder.n(2)),
	},
	{
		id: "sqrt",
		icon: "√x",
		label: "Square root",
		apply: (builder, expr) => builder.sqrt(expr),
	},
	{
		id: "recip",
		icon: "1/x",
		label: "Reciprocal",
		apply: (builder, expr) => builder.div(builder.n(1), expr),
	},
	{
		id: "sum",
		icon: "Σ",
		label: "Sum from i=1 to n",
		apply: (builder, expr) => builder.sum(expr),
	},
	{
		id: "int",
		icon: "∫",
		label: "Integrate over x",
		apply: (builder, expr) => builder.integral(expr, "x"),
	},
	{
		id: "deriv",
		icon: "d/dx",
		label: "Differentiate",
		apply: (builder, expr) => builder.deriv(expr, "x"),
	},
	{
		id: "add-y",
		icon: "+ y",
		label: "Add a term",
		apply: (builder, expr) => builder.add(expr, builder.v("y")),
	},
	{
		id: "mul-y",
		icon: "· y",
		label: "Multiply by",
		apply: (builder, expr) => builder.mul(expr, builder.v("y")),
	},
	{
		id: "log",
		icon: "log",
		label: "Logarithm",
		apply: (builder, expr) => builder.log(expr),
	},
	{
		id: "lim",
		icon: "lim",
		label: "Limit as x→∞",
		apply: (builder, expr) => builder.limit(expr, "x", "\\infty"),
	},
	{
		id: "neg",
		icon: "−",
		label: "Negate",
		apply: (builder, expr) => builder.neg(expr),
	},
];
