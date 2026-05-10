import { Colors, Controls, type FlumeConfig } from "#/components/flume";

/*
registerPorts adds the base port types used by all compute nodes.

- tensor: the universal data flow port (operations and activations)
- optimizer_state: connects optimizer nodes to operation nodes
*/
export function registerPorts(config: FlumeConfig): FlumeConfig {
	return config
		.addPortType({
			type: "tensor",
			name: "tensor",
			label: "Tensor",
			color: Colors.yellow,
			controls: [],
		})
		.addPortType({
			type: "optimizer_state",
			name: "optimizer_state",
			label: "Optimizer",
			color: Colors.purple,
			controls: [],
		})
		.addPortType({
			type: "config_float",
			name: "config_float",
			label: "Float",
			color: Colors.grey,
			hidePort: true,
			controls: [
				Controls.number({
					name: "value",
					label: "Value",
					defaultValue: 0,
				}),
			],
		})
		.addPortType({
			type: "config_int",
			name: "config_int",
			label: "Int",
			color: Colors.grey,
			hidePort: true,
			controls: [
				Controls.number({
					name: "value",
					label: "Value",
					defaultValue: 0,
				}),
			],
		})
		.addPortType({
			type: "config_bool",
			name: "config_bool",
			label: "Bool",
			color: Colors.grey,
			hidePort: true,
			controls: [
				Controls.checkbox({
					name: "value",
					label: "Value",
					defaultValue: false,
				}),
			],
		});
}

/*
portTypeForKind maps a schema kind to the Flume port type used for
its data-flow inputs and outputs.
*/
export function portTypeForKind(kind: string): string {
	return kind === "Optimizer" ? "optimizer_state" : "tensor";
}
