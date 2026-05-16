import ContextMenu from "#/components/flume/ContextMenu/ContextMenu";
import { type Schema, useOperations } from "#/service/compute";
import { NODE_TYPES, type PortDef, type PortType } from "./types";

interface NodePalettePick {
	typeId: number;
	opId?: string;
	inputs?: PortDef[];
	outputs?: PortDef[];
}

interface NodePaletteProps {
	x: number;
	y: number;
	onPick: (pick: NodePalettePick) => void;
	onClose: () => void;
}

const CONVERTER_TYPE_ID = NODE_TYPES.findIndex(
	(type) => type.kind === "converter",
);

/*
normalizePortType maps the YAML manifests' richer type vocabulary (float,
int, scalar, int[], list) onto the frontend's small fixed palette. Unknown
values collapse to "any" so an unrecognized type doesn't break placement.
*/
function normalizePortType(raw: string): PortType {
	switch (raw) {
		case "tensor":
		case "string":
		case "bool":
		case "number":
		case "trigger":
		case "any":
			return raw;
		case "float":
		case "int":
		case "scalar":
			return "number";
		default:
			return "any";
	}
}

function schemaToPortDefs(ports: Schema["inputs"]): PortDef[] {
	return ports.map((port) => ({
		name: port.name,
		type: normalizePortType(port.type),
	}));
}

/*
NodePalette wraps flume's ContextMenu and merges two data sources:
  1. The local NODE_TYPES registry (default / source / sink / scalar /
     gate / converter), shown under "Node types".
  2. Operation schemas fetched from GET /backend/compute/operation,
     grouped by their YAML `category` field.

Selecting a local type emits `{ typeId }`. Selecting a backend op emits
`{ typeId: 0 (default placeholder), opId, inputs, outputs }` so the
placed node carries the op's identity and port shapes without needing
a registry entry per op.
*/
export function NodePalette({ x, y, onPick, onClose }: NodePaletteProps) {
	const { data: ops } = useOperations();

	const localOptions = NODE_TYPES.map((type, idx) => ({
		category: "Node types",
		description: `${type.inputs.length} in · ${type.outputs.length} out`,
		label: type.label,
		value: `type:${idx}`,
	}));

	const opOptions = ops
		? Object.values(ops).map((schema) => ({
				category: schema.category || "Other",
				description: schema.description,
				label: schema.label || schema.name || schema.op,
				value: `op:${schema.op}`,
			}))
		: [];

	const options = [...localOptions, ...opOptions];

	return (
		<ContextMenu
			emptyText="No operations available."
			label="Add node"
			onOptionSelected={(option) => {
				if (option.value.startsWith("type:")) {
					onPick({ typeId: Number(option.value.slice("type:".length)) });
					return;
				}
				if (option.value.startsWith("op:") && ops) {
					const opId = option.value.slice("op:".length);
					const schema = ops[opId];
					if (!schema) return;
					onPick({
						inputs: schemaToPortDefs(schema.inputs),
						opId,
						outputs: schemaToPortDefs(schema.outputs),
						typeId: 0,
					});
				}
			}}
			onRequestClose={onClose}
			options={options}
			x={x}
			y={y}
		/>
	);
}

export { CONVERTER_TYPE_ID };
