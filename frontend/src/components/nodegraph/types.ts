/*
Node-type registry and port-type palette. The GPU encodes a port type as a
small integer (0..PORT_TYPES.length-1); the shader looks the color up in a
uniform palette. To add a new port type: append to PORT_TYPES, append the
color to PORT_COLORS, done — no shader edits.
*/

export const PORT_TYPES = [
	"any",
	"number",
	"string",
	"tensor",
	"bool",
	"trigger",
] as const;

export type PortType = (typeof PORT_TYPES)[number];

export const PORT_COLORS: Record<PortType, [number, number, number]> = {
	any: [0.55, 0.58, 0.65],
	number: [0.38, 0.72, 0.92],
	string: [0.92, 0.68, 0.36],
	tensor: [0.68, 0.42, 0.92],
	bool: [0.42, 0.85, 0.55],
	trigger: [0.95, 0.35, 0.45],
};

export interface PortDef {
	name: string;
	type: PortType;
}

export interface NodeType {
	kind: string;
	label: string;
	inputs: PortDef[];
	outputs: PortDef[];
}

/*
Built-in node types. typeId is the index into this array — the GPU's
nodesTexture stores it in the .b channel of each node texel.
*/
export const NODE_TYPES: NodeType[] = [
	{
		kind: "default",
		label: "Node",
		inputs: [{ name: "in", type: "any" }],
		outputs: [{ name: "out", type: "any" }],
	},
	{
		kind: "source",
		label: "Source",
		inputs: [],
		outputs: [{ name: "value", type: "tensor" }],
	},
	{
		kind: "sink",
		label: "Sink",
		inputs: [{ name: "value", type: "tensor" }],
		outputs: [],
	},
	{
		kind: "scalar",
		label: "Scalar",
		inputs: [
			{ name: "x", type: "number" },
			{ name: "y", type: "number" },
		],
		outputs: [
			{ name: "sum", type: "number" },
			{ name: "diff", type: "number" },
		],
	},
	{
		kind: "gate",
		label: "Gate",
		inputs: [
			{ name: "in", type: "tensor" },
			{ name: "open", type: "bool" },
		],
		outputs: [{ name: "out", type: "tensor" }],
	},
	{
		// Bridges between strictly-typed ports. Default config is identity
		// (number→number); per-node overrides on Node.inputs/Node.outputs
		// let the user reconfigure each placed converter independently.
		kind: "converter",
		label: "Converter",
		inputs: [{ name: "in", type: "number" }],
		outputs: [{ name: "out", type: "number" }],
	},
];

export function getNodeType(typeId: number): NodeType {
	return NODE_TYPES[typeId] ?? NODE_TYPES[0];
}

export function portTypeId(type: PortType): number {
	return PORT_TYPES.indexOf(type);
}

/*
Strict type equality — only same-typed ports may connect. There is no
implicit widening through "any"; an "any" port only connects to another
"any" port. The output→input direction is enforced separately.
*/
export function portsCompatible(a: PortType, b: PortType): boolean {
	return a === b;
}
