/*
MathSubselectionAst: immutable expression trees for clickable KaTeX subselection,
ported from docs/math_subselection_developed_demo.html.
*/

export type AstNode =
	| VarNode
	| NumNode
	| PowNode
	| SqrtNode
	| DivNode
	| SumNode
	| IntegralNode
	| DerivNode
	| AddNode
	| MulNode
	| LogNode
	| LimitNode
	| NegNode;

export type VarNode = { _id: number; type: "var"; name: string };
export type NumNode = { _id: number; type: "num"; value: number };
export type PowNode = {
	_id: number;
	type: "pow";
	base: AstNode;
	exp: AstNode;
};
export type SqrtNode = { _id: number; type: "sqrt"; arg: AstNode };
export type DivNode = {
	_id: number;
	type: "div";
	num: AstNode;
	den: AstNode;
};
export type SumNode = { _id: number; type: "sum"; body: AstNode };
export type IntegralNode = {
	_id: number;
	type: "integral";
	body: AstNode;
	v: string;
};
export type DerivNode = {
	_id: number;
	type: "deriv";
	body: AstNode;
	v: string;
};
export type AddNode = { _id: number; type: "add"; a: AstNode; b: AstNode };
export type MulNode = { _id: number; type: "mul"; a: AstNode; b: AstNode };
export type LogNode = { _id: number; type: "log"; arg: AstNode };
export type LimitNode = {
	_id: number;
	type: "limit";
	body: AstNode;
	v: string;
	to: string;
};
export type NegNode = { _id: number; type: "neg"; arg: AstNode };

const CHILD_KEYS = [
	"base",
	"exp",
	"arg",
	"num",
	"den",
	"body",
	"a",
	"b",
] as const;

export function cloneAst(node: AstNode): AstNode {
	return JSON.parse(JSON.stringify(node)) as AstNode;
}

export function prec(node: AstNode): number {
	if (node.type === "add") {
		return 1;
	}

	if (node.type === "mul") {
		return 2;
	}

	if (node.type === "neg") {
		return 3;
	}

	if (
		node.type === "sum" ||
		node.type === "integral" ||
		node.type === "deriv" ||
		node.type === "limit"
	) {
		return 1;
	}

	return 5;
}

export function atomicNode(node: AstNode): boolean {
	return node.type === "var" || node.type === "num";
}

export class MathAstBuilder {
	private nextId = 1;

	reset(): void {
		this.nextId = 1;
	}

	private id(): number {
		return this.nextId++;
	}

	v(name: string): VarNode {
		return { _id: this.id(), type: "var", name };
	}

	n(value: number): NumNode {
		return { _id: this.id(), type: "num", value };
	}

	pow(base: AstNode, exp: AstNode): PowNode {
		return { _id: this.id(), type: "pow", base, exp };
	}

	sqrt(arg: AstNode): SqrtNode {
		return { _id: this.id(), type: "sqrt", arg };
	}

	div(num: AstNode, den: AstNode): DivNode {
		return { _id: this.id(), type: "div", num, den };
	}

	sum(body: AstNode): SumNode {
		return { _id: this.id(), type: "sum", body };
	}

	integral(body: AstNode, integrationVar: string): IntegralNode {
		return { _id: this.id(), type: "integral", body, v: integrationVar };
	}

	deriv(body: AstNode, derivVar: string): DerivNode {
		return { _id: this.id(), type: "deriv", body, v: derivVar };
	}

	add(a: AstNode, b: AstNode): AddNode {
		return { _id: this.id(), type: "add", a, b };
	}

	mul(a: AstNode, b: AstNode): MulNode {
		return { _id: this.id(), type: "mul", a, b };
	}

	log(arg: AstNode): LogNode {
		return { _id: this.id(), type: "log", arg };
	}

	limit(body: AstNode, limitVar: string, to: string): LimitNode {
		return { _id: this.id(), type: "limit", body, v: limitVar, to };
	}

	neg(arg: AstNode): NegNode {
		return { _id: this.id(), type: "neg", arg };
	}
}

export type PresetKey = "default" | "variance" | "gauss" | "atom";

export function buildPreset(builder: MathAstBuilder, key: PresetKey): AstNode {
	builder.reset();

	switch (key) {
		case "default":
			return builder.add(
				builder.pow(builder.v("x"), builder.n(2)),
				builder.v("y"),
			);
		case "variance":
			return builder.sum(
				builder.pow(
					builder.add(builder.v("x_i"), builder.neg(builder.v("\\bar{x}"))),
					builder.n(2),
				),
			);
		case "gauss":
			return builder.pow(
				builder.v("e"),
				builder.neg(
					builder.div(builder.pow(builder.v("x"), builder.n(2)), builder.n(2)),
				),
			);
		case "atom":
			return builder.v("x");
	}
}

function powBaseLatex(node: AstNode, fn: (child: AstNode) => string): string {
	const rendered = fn(node);
	return atomicNode(node) ? rendered : `\\left(${rendered}\\right)`;
}

function wrapLatex(
	node: AstNode,
	minPrec: number,
	fn: (child: AstNode) => string,
): string {
	const rendered = fn(node);
	return prec(node) < minPrec ? `\\left(${rendered}\\right)` : rendered;
}

export function innerPlain(node: AstNode): string {
	switch (node.type) {
		case "var":
			return node.name;
		case "num":
			return String(node.value);
		case "pow":
			return `${powBaseLatex(node.base, innerPlain)}^{${innerPlain(node.exp)}}`;
		case "sqrt":
			return `\\sqrt{${innerPlain(node.arg)}}`;
		case "div":
			return `\\frac{${innerPlain(node.num)}}{${innerPlain(node.den)}}`;
		case "sum":
			return `\\sum_{i=1}^{n} ${wrapLatex(node.body, 2, innerPlain)}`;
		case "integral":
			return `\\int ${wrapLatex(node.body, 2, innerPlain)} \\, d${node.v}`;
		case "deriv":
			return `\\frac{d}{d${node.v}}${wrapLatex(node.body, 3, innerPlain)}`;
		case "add":
			return `${innerPlain(node.a)} + ${innerPlain(node.b)}`;
		case "mul":
			return `${wrapLatex(node.a, 2, innerPlain)} \\cdot ${wrapLatex(node.b, 2, innerPlain)}`;
		case "log":
			return `\\log\\left(${innerPlain(node.arg)}\\right)`;
		case "limit":
			return `\\lim_{${node.v} \\to ${node.to}} ${wrapLatex(node.body, 2, innerPlain)}`;
		case "neg":
			return `-${wrapLatex(node.arg, 3, innerPlain)}`;
	}
}

function toLatexInteractive(node: AstNode): string {
	return `\\htmlClass{ast-node node-${node._id}}{${innerInteractive(node)}}`;
}

function innerInteractive(node: AstNode): string {
	switch (node.type) {
		case "var":
			return node.name;
		case "num":
			return String(node.value);
		case "pow":
			return `${powBaseLatex(node.base, toLatexInteractive)}^{${toLatexInteractive(node.exp)}}`;
		case "sqrt":
			return `\\sqrt{${toLatexInteractive(node.arg)}}`;
		case "div":
			return `\\frac{${toLatexInteractive(node.num)}}{${toLatexInteractive(node.den)}}`;
		case "sum":
			return `\\sum_{i=1}^{n} ${wrapLatex(node.body, 2, toLatexInteractive)}`;
		case "integral":
			return `\\int ${wrapLatex(node.body, 2, toLatexInteractive)} \\, d${node.v}`;
		case "deriv":
			return `\\frac{d}{d${node.v}}${wrapLatex(node.body, 3, toLatexInteractive)}`;
		case "add":
			return `${toLatexInteractive(node.a)} + ${toLatexInteractive(node.b)}`;
		case "mul":
			return `${wrapLatex(node.a, 2, toLatexInteractive)} \\cdot ${wrapLatex(node.b, 2, toLatexInteractive)}`;
		case "log":
			return `\\log\\left(${toLatexInteractive(node.arg)}\\right)`;
		case "limit":
			return `\\lim_{${node.v} \\to ${node.to}} ${wrapLatex(node.body, 2, toLatexInteractive)}`;
		case "neg":
			return `-${wrapLatex(node.arg, 3, toLatexInteractive)}`;
	}
}

export function rootToInteractiveKatex(root: AstNode): string {
	return toLatexInteractive(root);
}

export function findNodeById(node: AstNode, id: number): AstNode | null {
	if (node._id === id) {
		return node;
	}

	for (const key of CHILD_KEYS) {
		const child = node[key as keyof AstNode] as AstNode | undefined;

		if (child && typeof child === "object" && "_id" in child) {
			const found = findNodeById(child, id);

			if (found) {
				return found;
			}
		}
	}

	return null;
}

export function replaceNodeById(
	node: AstNode,
	id: number,
	replace: (matched: AstNode) => AstNode,
): AstNode {
	if (node._id === id) {
		return replace(node);
	}

	const next = { ...node } as AstNode;

	for (const key of CHILD_KEYS) {
		const child = node[key as keyof AstNode] as AstNode | undefined;

		if (child && typeof child === "object" && "_id" in child) {
			(next as unknown as Record<string, AstNode>)[key] = replaceNodeById(
				child,
				id,
				replace,
			);
		}
	}

	return next;
}

export function normalizeLatexComparable(source: string): string {
	return source
		.trim()
		.replace(/\s+/g, " ")
		.replace(/ \+ /g, " + ")
		.replace(/ = /g, " = ");
}

export function describeAstNode(node: AstNode): string {
	switch (node.type) {
		case "var":
			return `variable "${node.name}"`;
		case "num":
			return `the number ${node.value}`;
		case "pow":
			return "a power";
		case "sqrt":
			return "a square root";
		case "div":
			return "a fraction";
		case "sum":
			return "a sum";
		case "integral":
			return "an integral";
		case "deriv":
			return "a derivative";
		case "add":
			return "a sum of terms";
		case "mul":
			return "a product";
		case "log":
			return "a logarithm";
		case "limit":
			return "a limit";
		case "neg":
			return "a negation";
		default:
			return "a subexpression";
	}
}

export function matchPresetFromLatex(latex: string): AstNode | null {
	const target = normalizeLatexComparable(latex);

	if (target === "") {
		return null;
	}

	const builder = new MathAstBuilder();
	const keys: PresetKey[] = ["default", "variance", "gauss", "atom"];

	for (const key of keys) {
		const candidate = buildPreset(builder, key);
		const asPlain = normalizeLatexComparable(innerPlain(candidate));

		if (asPlain === target) {
			return cloneAst(candidate);
		}
	}

	return null;
}
