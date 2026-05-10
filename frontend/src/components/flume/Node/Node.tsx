"use client";

import { ChevronDownIcon } from "lucide-react";
import type { RefObject } from "react";
import React from "react";
import { createPortal } from "react-dom";
import { BarVertical } from "#/components/charts/bar-vertical";
import {
	calculateEdgePath,
	collectDomObstacleRects,
	getPortRect,
} from "#/components/flume/connectionCalculator";
import { CONNECTIONS_ID } from "#/components/flume/constants";
import {
	CacheContext,
	EditorIdContext,
	NodeDispatchContext,
	NodeTypesContext,
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
}: NodeProps) => {
	const cache = React.useContext(CacheContext) ?? undefined;
	const nodeTypes = React.useContext(NodeTypesContext) ?? {};
	const nodesDispatch = React.useContext(NodeDispatchContext);
	const editorId = React.useContext(EditorIdContext);
	const stageState = React.useContext(StageContext) ?? {
		scale: 0,
		translate: { x: 0, y: 0 },
	};
	const currentNodeType = nodeTypes[type];
	const edgeRouting = useEdgeRouting();
	const {
		label,
		deletable,
		description,
		inputs = [],
		outputs = [],
	} = currentNodeType;

	const nodeWrapper = React.useRef<HTMLDivElement>(null);
	const [menuOpen, setMenuOpen] = React.useState(false);
	const [menuCoordinates, setMenuCoordinates] = React.useState({
		x: 0,
		y: 0,
	});

	const byScale = (value: number) => (1 / stageState.scale) * value;

	const updateConnectionsByTransput = (
		transput: ConnectionMap = {},
		isOutput?: boolean,
	) => {
		Object.entries(transput).forEach(([portName, outputs]) => {
			outputs.forEach((output) => {
				const toRect = getPortRect(
					id,
					portName,
					isOutput ? "output" : "input",
					cache,
				);
				const fromRect = getPortRect(
					output.nodeId,
					output.portName,
					isOutput ? "input" : "output",
					cache,
				);
				const fromHalfW = (fromRect?.width ?? 0) / 2;
				const fromHalfH = (fromRect?.height ?? 0) / 2;
				const toHalfW = (toRect?.width ?? 0) / 2;
				const toHalfH = (toRect?.height ?? 0) / 2;
				let combined: string;
				if (isOutput) {
					combined = id + portName + output.nodeId + output.portName;
				} else {
					combined = output.nodeId + output.portName + id + portName;
				}
				let cnx: SVGPathElement | Connections | null;
				const cachedConnection = cache?.current?.connections[combined];
				if (cachedConnection) {
					cnx = cachedConnection;
				} else {
					cnx = document.querySelector<SVGPathElement>(
						`[data-connection-id="${combined}"]`,
					);
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
				const connStage = document
					.getElementById(`${CONNECTIONS_ID}${editorId}`)
					?.getBoundingClientRect();
				const obstaclesVertical =
					edgeRouting === "orthogonal" && connStage
						? collectDomObstacleRects(connStage, stageState.scale, new Set())
						: undefined;
				const obstaclesHorizontal =
					edgeRouting === "orthogonal" && connStage
						? collectDomObstacleRects(
								connStage,
								stageState.scale,
								new Set([id, output.nodeId]),
							)
						: undefined;
				cnx?.setAttribute(
					"d",
					calculateEdgePath(
						edgeRouting,
						isOutput ? to : from,
						isOutput ? from : to,
						obstaclesVertical,
						obstaclesHorizontal,
					),
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
			updateNodeConnections();
		}
	};

	const startDrag = () => {
		onDragStart();
	};

	const handleContextMenu = (e: MouseEvent | React.MouseEvent) => {
		e.preventDefault();
		e.stopPropagation();
		setMenuCoordinates({ x: e.clientX, y: e.clientY });
		setMenuOpen(true);
		return false;
	};

	const closeContextMenu = () => {
		setMenuOpen(false);
	};

	const deleteNode = () => {
		nodesDispatch?.({
			type: NodesActionType.REMOVE_NODE,
			nodeId: id,
		});
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

	const suppressEmbeddedPortControlPrep = React.useCallback(
		(e: React.MouseEvent<HTMLDivElement>) =>
			Boolean(
				e.target instanceof Element &&
					e.target.closest("button, input, textarea, select, option"),
			),
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

	return (
		<Draggable
			className="absolute left-0 top-0 cursor-default select-none"
			style={{
				width,
				transform: `translate(${x}px, ${y}px)`,
			}}
			onDragStart={startDrag}
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
			<Frame className="min-w-0 w-full max-w-xs">
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
												description:
													"Deletes a node and all of its connections.",
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
