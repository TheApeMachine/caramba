import { useCallback, useMemo, useRef, useState } from "react";
import { FlumeConfig, NodeEditor, type NodeMap } from "#/components/flume";
import type { EdgeRoutingMode } from "#/components/flume/connectionCalculator";
import type { GraphLayoutMode } from "#/components/flume/graphLayout";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import {
	Select,
	SelectItem,
	SelectPopup,
	SelectTrigger,
	SelectValue,
} from "#/components/ui/select";
import {
	useArchitectures,
	useBlocks,
	useLoadArchitecture,
	useOperations,
	useOptimizers,
	useSaveArchitecture,
} from "#/service/compute";
import { registerNodes } from "./nodes";
import { registerPorts } from "./ports";
import { STORE_SCHEMAS } from "./stores";

export type NodeGraphProps = {
	nodes?: NodeMap;
	onNodesChange?: (nodes: NodeMap) => void;
	context?: unknown;
};

export const NodeGraph = ({
	nodes: controlledNodes,
	onNodesChange,
	context,
}: NodeGraphProps) => {
	const editorRef = useRef<{ getNodes: () => NodeMap }>(null);
	const [internalNodes, setInternalNodes] = useState<NodeMap>({});
	const [edgeRoutingMode, setEdgeRoutingMode] =
		useState<EdgeRoutingMode>("smooth");
	const [graphLayoutMode, setGraphLayoutMode] =
		useState<GraphLayoutMode>("freeform");
	const [saveName, setSaveName] = useState("");
	const [loadName, setLoadName] = useState("");

	const { data: operations } = useOperations();
	const { data: optimizers } = useOptimizers();
	const { data: blocks } = useBlocks();
	const { data: architectureNames = [] } = useArchitectures();
	const { data: loadedArchitecture } = useLoadArchitecture(loadName);
	const saveArchitecture = useSaveArchitecture();

	const config = useMemo(() => {
		const cfg = registerPorts(new FlumeConfig());
		const allSchemas = { ...operations, ...optimizers, ...blocks, ...STORE_SCHEMAS };
		if (operations) registerNodes(cfg, operations, allSchemas);
		if (optimizers) registerNodes(cfg, optimizers, allSchemas);
		if (blocks) registerNodes(cfg, blocks, allSchemas);
		registerNodes(cfg, STORE_SCHEMAS, allSchemas);
		return cfg;
	}, [operations, optimizers, blocks]);

	const nodes = loadedArchitecture ?? controlledNodes ?? internalNodes;
	const editorKey = loadedArchitecture ? `loaded-${loadName}` : `__internal__-${loadName}`;

	const handleChange = useCallback(
		(next: NodeMap) => {
			onNodesChange?.(next);
			if (controlledNodes === undefined) setInternalNodes(next);
		},
		[controlledNodes, onNodesChange],
	);

	const handleSave = useCallback(() => {
		if (!saveName.trim()) return;
		const current = editorRef.current?.getNodes() ?? internalNodes;
		saveArchitecture.mutate({ name: saveName.trim(), nodes: current });
	}, [saveName, internalNodes, saveArchitecture]);

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
							if (next !== null && next !== edgeRoutingMode)
								setEdgeRoutingMode(next as EdgeRoutingMode);
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
						Layout
					</span>
					<Select
						onValueChange={(next) => {
							if (next !== null && next !== graphLayoutMode)
								setGraphLayoutMode(next as GraphLayoutMode);
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

				<Flex.Row align="center" gap={2}>
					<span className="whitespace-nowrap text-muted-foreground text-xs">
						Load
					</span>
					<Select
						onValueChange={(next) => { if (next) setLoadName(next); }}
						value={loadName}
					>
						<SelectTrigger className="min-w-36" size="sm">
							<SelectValue placeholder="Architecture…" />
						</SelectTrigger>
						<SelectPopup>
							{architectureNames.map((n) => (
								<SelectItem key={n} value={n}>{n}</SelectItem>
							))}
						</SelectPopup>
					</Select>
				</Flex.Row>

				<Flex.Row align="center" gap={2}>
					<span className="whitespace-nowrap text-muted-foreground text-xs">
						Save as
					</span>
					<Input
						className="h-7 w-32 text-xs"
						placeholder="name…"
						value={saveName}
						onChange={(e) => setSaveName(e.target.value)}
						onKeyDown={(e) => { if (e.key === "Enter") handleSave(); }}
					/>
				</Flex.Row>
			</Flex.Row>

			<Flex.Column className="min-h-[55dvh] min-w-0 flex-1" fullHeight fullWidth>
				<NodeEditor
					key={editorKey}
					ref={editorRef}
					portTypes={config.portTypes}
					nodeTypes={config.nodeTypes}
					nodes={nodes}
					context={context}
					onChange={handleChange}
					edgeRoutingMode={edgeRoutingMode}
					graphLayoutMode={graphLayoutMode}
				/>
			</Flex.Column>
		</Flex.Column>
	);
};
