import { useCallback, useState } from "react";
import { FlumeConfig, NodeEditor, type NodeMap } from "#/components/flume";
import type { EdgeRoutingMode } from "#/components/flume/connectionCalculator";
import type { GraphLayoutMode } from "#/components/flume/graphLayout";
import { Flex } from "#/components/ui/flex";
import {
	Select,
	SelectItem,
	SelectPopup,
	SelectTrigger,
	SelectValue,
} from "#/components/ui/select";

import { registerHuggingFaceNodes } from "./nodes";
import { registerHuggingFacePorts } from "./ports";

/** Hugging Face–oriented graph: chain `hf_text` between Text → HF Model nodes; root sink marks pipeline output. */
export const huggingFaceNodeGraphConfig = registerHuggingFaceNodes(
	registerHuggingFacePorts(new FlumeConfig()),
	[],
);

export type NodeGraphProps = {
	/** Controlled graph state. Omit to keep nodes in component state. */
	nodes?: NodeMap;
	onNodesChange?: (nodes: NodeMap) => void;
	context?: unknown;
	config?: FlumeConfig;
};

export const NodeGraph = ({
	nodes: controlledNodes,
	onNodesChange,
	context,
	config = huggingFaceNodeGraphConfig,
}: NodeGraphProps) => {
	const [internalNodes, setInternalNodes] = useState<NodeMap>({});
	const [edgeRoutingMode, setEdgeRoutingMode] =
		useState<EdgeRoutingMode>("smooth");
	const [graphLayoutMode, setGraphLayoutMode] =
		useState<GraphLayoutMode>("freeform");

	const nodes = controlledNodes ?? internalNodes;

	const handleChange = useCallback(
		(next: NodeMap) => {
			onNodesChange?.(next);
			if (controlledNodes === undefined) {
				setInternalNodes(next);
			}
		},
		[controlledNodes, onNodesChange],
	);

	return (
		<Flex.Column className="min-h-0 w-full flex-1" fullHeight fullWidth gap={2}>
			<Flex.Row
				align="center"
				className="shrink-0 rounded-xl border bg-muted/48 px-3 py-2"
				gap={4}
				wrap="wrap"
			>
				<Flex.Row align="center" className="min-w-44" gap={2}>
					<span className="whitespace-nowrap text-muted-foreground text-xs">
						Edges
					</span>
					<Select
						onValueChange={(next) => {
							if (next !== null && next !== edgeRoutingMode) {
								setEdgeRoutingMode(next as EdgeRoutingMode);
							}
						}}
						value={edgeRoutingMode}
					>
						<SelectTrigger className="w-full min-w-0" size="sm">
							<SelectValue />
						</SelectTrigger>
						<SelectPopup>
							<SelectItem value="smooth">Smooth curves</SelectItem>
							<SelectItem value="straight">Straight lines</SelectItem>
							<SelectItem value="orthogonal">Orthogonal (step)</SelectItem>
						</SelectPopup>
					</Select>
				</Flex.Row>
				<Flex.Row align="center" className="min-w-44" gap={2}>
					<span className="whitespace-nowrap text-muted-foreground text-xs">
						Node layout
					</span>
					<Select
						onValueChange={(next) => {
							if (next !== null && next !== graphLayoutMode) {
								setGraphLayoutMode(next as GraphLayoutMode);
							}
						}}
						value={graphLayoutMode}
					>
						<SelectTrigger className="w-full min-w-0" size="sm">
							<SelectValue />
						</SelectTrigger>
						<SelectPopup>
							<SelectItem value="freeform">Free-form</SelectItem>
							<SelectItem value="horizontalPipeline">
								Horizontal pipeline
							</SelectItem>
							<SelectItem value="verticalPipeline">
								Vertical pipeline
							</SelectItem>
						</SelectPopup>
					</Select>
				</Flex.Row>
			</Flex.Row>
			<Flex.Column
				className="min-h-[55dvh] min-w-0 flex-1"
				fullHeight
				fullWidth
			>
				<NodeEditor
					portTypes={config.portTypes}
					nodeTypes={config.nodeTypes}
					nodes={nodes}
					context={context}
					onChange={handleChange}
					edgeRoutingMode={edgeRoutingMode}
					graphLayoutMode={graphLayoutMode}
					defaultNodes={[
						{ type: "hf_pipeline_root", x: 260, y: -40 },
						{ type: "hf_text_source", x: -260, y: -80 },
						{ type: "hf_inference", x: 0, y: -60 },
					]}
				/>
			</Flex.Column>
		</Flex.Column>
	);
};
