"use client";

import { ChevronDownIcon, Maximize2Icon, Minimize2Icon, NetworkIcon } from "lucide-react";
import type { RefObject } from "react";
import React from "react";
import { createPortal } from "react-dom";
import { BarVertical } from "#/components/charts/bar-vertical";
import {
	calculateEdgePath,
	getPortRect,
} from "#/components/flume/connectionCalculator";

import {
	CacheContext,
	NodeDispatchContext,
	NodeMapContext,
	NodeTypesContext,
	PortTypesContext,
	RecalculateConnectionsWorkerContext,
	StageContext,
	useEdgeRouting,
} from "#/components/flume/context";
import { NodesActionType } from "#/components/flume/nodesReducer";
import type {
	ConnectionMap,
	Connections,
	Coordinate,
	InputData,
	NodeHeaderRenderCallback,
	NodeMap,
	SelectOption,
} from "#/components/flume/types";
import { Card, CardPanel } from "#/components/ui/card";
import {
	Collapsible,
	CollapsiblePanel,
	CollapsibleTrigger,
} from "#/components/ui/collapsible";
import { Flex } from "#/components/ui/flex";
import { Form } from "#/components/ui/form";
import {
	Frame,
	FrameDescription,
	FrameFooter,
	FrameHeader,
	FrameTitle,
} from "#/components/ui/frame";
import { Typography } from "#/components/ui/typography";
import { cn } from "@/lib/utils";
import ContextMenu from "../ContextMenu/ContextMenu";
import Draggable from "../Draggable/Draggable";
import IoPorts from "../IoPorts/IoPorts";

/* Lazy to avoid circular dep — NodeEditor imports Node */
const NodeEditor = React.lazy(
	() => import("../NodeEditor").then((m) => ({ default: m.NodeEditor })),
);

const SUBGRAPH_WIDTH = 560;
const SUBGRAPH_HEIGHT = 360;

interface NodeProps {
	id: string;
	width: number;
	x: number;
	y: number;
	stageRect: RefObject<DOMRect | undefined>;
	connections: Connections;
	type: string;
	inputData: InputData;
	onDragStart: () => void;
	renderNodeHeader?: NodeHeaderRenderCallback;
	root?: boolean;
	subGraph?: NodeMap;
}

const Node = ({
	id,
	width,
	x,
	y,
	stageRect,
	connections,
	type,
	inputData,
	root,
	onDragStart,
	renderNodeHeader,
	subGraph,
}: NodeProps) => {
	const cache = React.useContext(CacheContext) ?? undefined;
	const nodeTypes = React.useContext(NodeTypesContext) ?? {};
	const portTypes = React.useContext(PortTypesContext) ?? {};
	const nodesDispatch = React.useContext(NodeDispatchContext);
	const recalculateWorker = React.useContext(RecalculateConnectionsWorkerContext);
	const nodeMap = React.useContext(NodeMapContext);
	const stageState = React.useContext(StageContext) ?? {
		scale: 0,
		translate: { x: 0, y: 0 },
	};

	const currentNodeType = nodeTypes[type];
	const edgeRouting = useEdgeRouting();
	const isBlock = Boolean(
		currentNodeType?.defaultSubGraph ||
		currentNodeType?.category === "memory" ||
		String(currentNodeType?.type ?? "").startsWith("block."),
	);


	const {
		label,
		deletable,
		description,
		inputs = [],
		outputs = [],
	} = currentNodeType;

	const nodeWrapper = React.useRef<HTMLDivElement>(null);
	const [menuOpen, setMenuOpen] = React.useState(false);
	const [menuCoordinates, setMenuCoordinates] = React.useState({ x: 0, y: 0 });
	const [subGraphOpen, setSubGraphOpen] = React.useState(false);
	const [subGraphFullscreen, setSubGraphFullscreen] = React.useState(false);

	const byScale = (value: number) => (1 / stageState.scale) * value;

	const updateConnectionsByTransput = (
		transput: ConnectionMap = {},
		isOutput?: boolean,
	) => {
		Object.entries(transput).forEach(([portName, outputs]) => {
			outputs.forEach((output) => {
				const toRect = getPortRect(id, portName, isOutput ? "output" : "input", cache);
				const fromRect = getPortRect(output.nodeId, output.portName, isOutput ? "input" : "output", cache);
				const fromHalfW = (fromRect?.width ?? 0) / 2;
				const fromHalfH = (fromRect?.height ?? 0) / 2;
				const toHalfW = (toRect?.width ?? 0) / 2;
				const toHalfH = (toRect?.height ?? 0) / 2;
				const combined = isOutput
					? id + portName + output.nodeId + output.portName
					: output.nodeId + output.portName + id + portName;
				let cnx: SVGPathElement | Connections | null;
				const cachedConnection = cache?.current?.connections[combined];
				if (cachedConnection) {
					cnx = cachedConnection;
				} else {
					cnx = document.querySelector<SVGPathElement>(`[data-connection-id="${combined}"]`);
					if (cnx && cache && cache.current) {
						cache.current.connections[combined] = cnx;
					}
				}
				const sx = stageRect.current?.x ?? 0;
				const sy = stageRect.current?.y ?? 0;
				const sHw = (stageRect.current?.width ?? 0) / 2;
				const sHh = (stageRect.current?.height ?? 0) / 2;
				const tx = byScale(stageState.translate.x);
				const ty = byScale(stageState.translate.y);
				const from = {
					x: byScale((fromRect?.x ?? 0) - sx + fromHalfW - sHw) + tx,
					y: byScale((fromRect?.y ?? 0) - sy - sHh + fromHalfH) + ty,
				};
				const to = {
					x: byScale((toRect?.x ?? 0) - sx + toHalfW - sHw) + tx,
					y: byScale((toRect?.y ?? 0) - sy - sHh + toHalfH) + ty,
				};
				cnx?.setAttribute(
					"d",
					calculateEdgePath(edgeRouting, isOutput ? to : from, isOutput ? from : to),
				);
			});
		});
	};

	const updateNodeConnections = () => {
		if (connections) {
			updateConnectionsByTransput(connections.inputs);
			updateConnectionsByTransput(connections.outputs, true);
		}
	};

	const stopDrag = (_e: unknown, coordinates: Coordinate) => {
		nodesDispatch?.({
			type: NodesActionType.SET_NODE_COORDINATES,
			...coordinates,
			nodeId: id,
		});
	};

	const handleDrag = ({ x, y }: Coordinate) => {
		if (nodeWrapper.current) {
			nodeWrapper.current.style.transform = `translate(${x}px,${y}px)`;
			if (edgeRouting === "orthogonal" && recalculateWorker) {
				recalculateWorker(nodeMap);
			} else {
				updateNodeConnections();
			}
		}
	};

	const handleContextMenu = (e: MouseEvent | React.MouseEvent) => {
		e.preventDefault();
		e.stopPropagation();
		setMenuCoordinates({ x: e.clientX, y: e.clientY });
		setMenuOpen(true);
		return false;
	};

	const closeContextMenu = () => setMenuOpen(false);

	const deleteNode = () => {
		nodesDispatch?.({ type: NodesActionType.REMOVE_NODE, nodeId: id });
	};

	const handleMenuOption = ({ value }: SelectOption) => {
		switch (value) {
			case "deleteNode":
				deleteNode();
				break;
			default:
				return;
		}
	};

	const handleSubGraphChange = React.useCallback(
		(next: NodeMap) => {
			nodesDispatch?.({
				type: NodesActionType.SET_NODE_SUBGRAPH,
				nodeId: id,
				subGraph: next,
			});
		},
		[id, nodesDispatch],
	);

	const suppressEmbeddedPortControlPrep = React.useCallback(
		(e: React.MouseEvent<HTMLDivElement>) => {
			if (!(e.target instanceof Element)) return false;
			if (e.target.closest("button, input, textarea, select, option")) return true;
			// Suppress only when the click originates inside the nested sub-graph
			// editor, not the outer stage that the block node itself lives in.
			const subgraphContainer = e.currentTarget.querySelector("[data-subgraph-editor]");
			return Boolean(subgraphContainer?.contains(e.target));
		},
		[],
	);

	const portalContainer =
		typeof document !== "undefined" ? document.body : null;

	const chartPreviewData = React.useMemo(() => {
		let h = 0;
		for (let i = 0; i < id.length; i++) {
			h = (h + id.charCodeAt(i) * (i + 1)) % 101;
		}
		return [
			{ label: "Load", value: 32 + (h % 45) },
			{ label: "Mem", value: 24 + ((h * 3) % 38) },
			{ label: "Ops", value: 40 + ((h * 7) % 34) },
		];
	}, [id]);

	const resolvedSubGraph = subGraph ?? currentNodeType?.defaultSubGraph;

	const subGraphEditor = resolvedSubGraph !== undefined && subGraphOpen ? (
		<React.Suspense fallback={null}>
			<NodeEditor
				nodes={resolvedSubGraph}
				nodeTypes={nodeTypes}
				portTypes={portTypes}
				onChange={handleSubGraphChange}
				disableComments
				disableFocusCapture
				className="rounded-lg border border-border/48 bg-background/80"
				style={{
					width: SUBGRAPH_WIDTH,
					height: SUBGRAPH_HEIGHT,
					pointerEvents: "all",
				}}
			/>
		</React.Suspense>
	) : null;

	const fullscreenOverlay =
		resolvedSubGraph !== undefined && subGraphOpen && subGraphFullscreen && portalContainer
			? createPortal(
					<div className="fixed inset-0 z-50 flex flex-col bg-background">
						<div className="flex items-center gap-3 border-b px-4 py-2 text-sm text-muted-foreground">
							<NetworkIcon className="size-4" />
							<span className="font-medium text-foreground">{label}</span>
							<span className="flex-1">{description}</span>
							<button
								type="button"
								onClick={() => setSubGraphFullscreen(false)}
								className="ml-auto flex items-center gap-1.5 rounded px-2 py-1 hover:bg-muted/60"
							>
								<Minimize2Icon className="size-4" />
								Exit full screen
							</button>
						</div>
						<div className="min-h-0 flex-1">
							<React.Suspense fallback={null}>
								<NodeEditor
									nodes={resolvedSubGraph}
									nodeTypes={nodeTypes}
									portTypes={portTypes}
									onChange={handleSubGraphChange}
									disableComments
									className="h-full w-full"
								/>
							</React.Suspense>
						</div>
					</div>,
					portalContainer,
				)
			: null;

	const nodeWidth = subGraphOpen && !subGraphFullscreen
		? Math.max(width, SUBGRAPH_WIDTH + 32)
		: width;

	return (
		<Draggable
			className="absolute left-0 top-0 cursor-default select-none"
			style={{
				width: nodeWidth,
				transform: `translate(${x}px, ${y}px)`,
			}}
			onDragStart={onDragStart}
			onDrag={handleDrag}
			onDragEnd={stopDrag}
			innerRef={nodeWrapper}
			data-node-id={id}
			data-flume-component="node"
			data-flume-node-type={currentNodeType.type}
			data-flume-component-is-root={!!root}
			onContextMenu={handleContextMenu}
			suppressDragPrep={suppressEmbeddedPortControlPrep}
			stageState={stageState}
			stageRect={stageRect}
		>
			<Frame className="min-w-0 w-full">
				<FrameHeader className="gap-2 py-3">
					<Collapsible defaultOpen={false}>
						<CollapsibleTrigger
							className={cn(
								"w-full cursor-pointer rounded-lg border border-border/72 bg-background/52 px-3 py-2 text-start font-medium text-sm outline-none transition-colors hover:bg-muted/52 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background [&[data-panel-open]_svg:last-child]:rotate-180",
							)}
							type="button"
						>
							<Flex.Row align="center" fullWidth justify="between" gap={2}>
								<span className="min-w-0 truncate text-muted-foreground">
									Node preview chart
								</span>
								<ChevronDownIcon
									aria-hidden
									className="size-4 shrink-0 text-muted-foreground transition-transform duration-200"
								/>
							</Flex.Row>
						</CollapsibleTrigger>
						<CollapsiblePanel className="mt-2">
							<div className="rounded-lg border border-border/48 bg-muted/24 px-2 py-2">
								<BarVertical data={chartPreviewData} height={128} />
							</div>
						</CollapsiblePanel>
					</Collapsible>
				</FrameHeader>

				<Card>
					<CardPanel>
						<Form>
							<Flex.Column fullWidth gap={4}>
								<IoPorts
									nodeId={id}
									inputs={inputs}
									outputs={outputs}
									connections={connections}
									updateNodeConnections={updateNodeConnections}
									inputData={inputData}
								/>
							</Flex.Column>
						</Form>
					</CardPanel>
				</Card>

				{isBlock && (
					<div className="border-t border-border/48 px-3 py-2">
						<Flex.Row align="center" gap={2}>
							<button
								type="button"
								onClick={() => setSubGraphOpen((v) => !v)}
								className="flex flex-1 items-center gap-1.5 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-muted/60 hover:text-foreground"
							>
								<NetworkIcon className="size-3.5" />
								{subGraphOpen ? "Collapse operations" : "Expand operations"}
							</button>
							{subGraphOpen && (
								<button
									type="button"
									onClick={() => setSubGraphFullscreen(true)}
									className="flex items-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-muted/60 hover:text-foreground"
								>
									<Maximize2Icon className="size-3.5" />
									Full screen
								</button>
							)}
						</Flex.Row>
						{subGraphOpen && !subGraphFullscreen && (
							<div
								className="mt-2"
								data-subgraph-editor
								style={{ pointerEvents: "all" }}
							>
								{subGraphEditor}
							</div>
						)}
					</div>
				)}

				<FrameFooter>
					<Flex.Column fullWidth gap={3}>
						<Flex.Column className="min-w-0 w-full" gap={1} fullWidth>
							{renderNodeHeader ? (
								renderNodeHeader(FrameTitle, currentNodeType, {
									openMenu: handleContextMenu,
									closeMenu: closeContextMenu,
									deleteNode,
								})
							) : (
								<>
									<FrameTitle data-flume-component="node-header">
										{label}
									</FrameTitle>
									<FrameDescription data-flume-component="node-description">
										<Typography.Small variant="muted" truncate>
											{description ?? ""}
										</Typography.Small>
									</FrameDescription>
								</>
							)}
						</Flex.Column>
					</Flex.Column>
				</FrameFooter>
			</Frame>

			{fullscreenOverlay}

			{portalContainer && menuOpen
				? createPortal(
						<ContextMenu
							x={menuCoordinates.x}
							y={menuCoordinates.y}
							options={[
								...(deletable !== false
									? [
											{
												label: "Delete Node",
												value: "deleteNode",
												description: "Deletes a node and all of its connections.",
											},
										]
									: []),
							]}
							onRequestClose={closeContextMenu}
							onOptionSelected={handleMenuOption}
							hideFilter
							label="Node Options"
							emptyText="This node has no options."
						/>,
						portalContainer,
					)
				: null}
		</Draggable>
	);
};

export default Node;
