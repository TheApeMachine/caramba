import type { FlumeConfig, PortType, PortTypeBuilder } from "#/components/flume";

type Operation = {
	name: string;
	label: string;
	description: string;
	initialWidth: number;
	inputs: (ports: { [portType: string]: PortTypeBuilder }) => PortType[];
	outputs: (ports: { [portType: string]: PortTypeBuilder }) => PortType[];
}

/**
 * Node types: literal text → one or more HF model nodes → pipeline root sink.
 * Connect `hf_text` outputs to `hf_inference` inputs to chain models (e.g. summarize → translate).
 */
export const registerHuggingFaceNodes = (config: FlumeConfig, operations: Operation[]): FlumeConfig => {
	operations.forEach(operation => {
		config.addNodeType({
			type: operation.name,
			label: operation.label,
			description: operation.description,
			initialWidth: operation.initialWidth,
			inputs: operation.inputs,
			outputs: operation.outputs,
		})
	})

	return config
		.addNodeType({
			type: "activation",
			label: "Activation",
			description: "Activation function",
			category: "Compute",
			initialWidth: 220,
			inputs: (ports) => [
				ports.hf_text({
					name: "input",
					label: "Input",
				}),
				ports.activation({
					name: "activation",
					label: "Activation",
				}),
			],
			outputs: (ports) => [
				ports.hf_text({
					name: "output",
					label: "Output",
				}),
			],
		})
		.addNodeType({
			type: "hf_text_source",
			label: "Text",
			description: "Literal text or upstream chain input surface",
			category: "Hugging Face",
			initialWidth: 220,
			inputs: [],
			outputs: (ports) => [
				ports.hf_text({
					name: "out",
					label: "Out",
				}),
			],
		})
		.addNodeType({
			type: "hf_inference",
			label: "HF Model",
			description:
				"Hugging Face model call (wire Text in from another node or type here when disconnected)",
			category: "Hugging Face",
			initialWidth: 280,
			inputs: (ports) => [
				ports.hf_model_settings(),
				ports.hf_inference_task(),
				ports.hf_text({
					name: "input",
					label: "Input",
				}),
			],
			outputs: (ports) => [
				ports.hf_text({
					name: "output",
					label: "Output",
				}),
			],
		})
		.addRootNodeType({
			type: "hf_pipeline_root",
			label: "Pipeline output",
			description: "Exactly one sink for the runnable pipeline graph",
			category: "Hugging Face",
			initialWidth: 220,
			inputs: (ports) => [
				ports.hf_text({
					name: "result",
					label: "Result",
				}),
			],
			outputs: [],
		});
}
