import { Colors, Controls, type FlumeConfig } from "#/components/flume";

/** Pipeline tasks you’d send to Inference API or a compatible backend (values are stable identifiers). */
export const HF_INFERENCE_TASK_OPTIONS = [
	{ label: "Text generation", value: "text-generation" },
	{ label: "Text2text generation", value: "text2text-generation" },
	{ label: "Summarization", value: "summarization" },
	{ label: "Translation", value: "translation" },
	{ label: "Fill-mask", value: "fill-mask" },
	{ label: "Feature extraction", value: "feature-extraction" },
	{ label: "Zero-shot classification", value: "zero-shot-classification" },
] as const;

/**
 * Port types for Hugging Face–style pipelines.
 * - `hf_text`: chain prompts / completions between nodes (green — same idea as string ports in Flume docs).
 * - `hf_model_settings`: model repo id as inline control (hidePort — no socket).
 * - `hf_inference_task`: pipeline task selector (hidePort).
 */
export function registerHuggingFacePorts(config: FlumeConfig): FlumeConfig {
	return config
		.addPortType({
			type: "weights",
			name: "weights",
			label: "Weights",
			color: Colors.yellow,
		})
		.addPortType({
			type: "activation",
			name: "activation",
			label: "Activation",
			color: Colors.blue,
			hidePort: true,
			controls: [
				Controls.select({
					name: "activation",
					label: "Activation",
					defaultValue: "relu",
					options: [
						{ label: "ReLU", value: "relu" },
						{ label: "GELU", value: "gelu" },
						{ label: "SELU", value: "selu" },
						{ label: "Sigmoid", value: "sigmoid" },
						{ label: "Swish", value: "swish" },
					],
				}),
			],
		})
		.addPortType({
			type: "hf_text",
			name: "hf_text",
			label: "Text",
			color: Colors.green,
			controls: [
				Controls.text({
					name: "text",
					label: "Text",
					defaultValue: "",
				}),
			],
		})
		.addPortType({
			type: "hf_model_settings",
			name: "hf_model_settings",
			label: "Model",
			color: Colors.purple,
			hidePort: true,
			controls: [
				Controls.text({
					name: "repo_id",
					label: "Model repo",
					defaultValue: "gpt2",
				}),
			],
		})
		.addPortType({
			type: "hf_inference_task",
			name: "hf_inference_task",
			label: "Task",
			color: Colors.grey,
			hidePort: true,
			controls: [
				Controls.select({
					name: "task",
					label: "Task",
					defaultValue: HF_INFERENCE_TASK_OPTIONS[0].value,
					options: [...HF_INFERENCE_TASK_OPTIONS],
				}),
			],
		});
}
