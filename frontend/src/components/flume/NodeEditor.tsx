import React, { useId } from "react";
import Cache from "#/components/flume/Cache";
import Comment from "#/components/flume/Comment/Comment";
import Connections from "#/components/flume/Connections/Connections";
import commentsReducer from "#/components/flume/commentsReducer";
import {
	createConnections,
	type EdgeRoutingMode,
} from "#/components/flume/connectionCalculator";
import {
	DRAG_CONNECTION_ID,
	PORT_LAYER_ID,
	STAGE_ID,
} from "#/components/flume/constants";
import Node from "#/components/flume/Node/Node";
import Stage from "#/components/flume/Stage/Stage";
import { portLayoutKey } from "#/components/flume/spatial-index";
import { useFlumeGraphWorker } from "#/components/flume/useFlumeGraphWorker";
import {
	ObstacleIndexContext,
	PortLayoutRegistrationContext,
	useSpatialIndex,
} from "#/components/flume/useSpatialIndex";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";
import usePrevious from "#/hooks/usePrevious";
import { cn } from "@/lib/utils";
import {
	CacheContext,
	ConnectionRecalculateContext,
	ContextContext,
	EdgeRoutingContext,
	EditorIdContext,
	FlumeGraphWorkerContext,
	NodeDispatchContext,
	NodeDragOverrideContext,
	NodeMapContext,
	NodeTypesContext,
	PortTypesContext,
	RecalculateStageRectContext,
	StageContext,
} from "./context";
import { dispatchGraphLayout, type GraphLayoutMode } from "./graphLayout";
import nodesReducer, {
	connectNodesReducer,
	getInitialNodes,
	NodesActionType,
} from "./nodesReducer";
import stageReducer from "./stageReducer";
import styles from "./styles.module.css";
import { dispatchFlumeToastAction, type ToastAction } from "./toastsReducer";
import type {
	CircularBehavior,
	DefaultConnection,
	DefaultNode,
	FlumeCommentMap,
	NodeHeaderRenderCallback,
	NodeMap,
	NodeTypeMap,
	PortTypeMap,
} from "./types";

const defaultContext = {};

export type NodeEditorHandle = {
	getNodes: () => NodeMap;
	getComments: () => FlumeCommentMap;
};

interface NodeEditorProps {
	ref?: React.Ref<NodeEditorHandle>;
	comments?: FlumeCommentMap;
	nodes?: NodeMap;
	nodeTypes: NodeTypeMap;
	portTypes: PortTypeMap;
	defaultNodes?: DefaultNode[];
	defaultConnections?: DefaultConnection[];
	context?: unknown;
	onChange?: (nodes: NodeMap) => void;
	onCommentsChange?: (comments: FlumeCommentMap) => void;
	initialScale?: number;
	spaceToPan?: boolean;
	hideComments?: boolean;
	disableComments?: boolean;
	disableZoom?: boolean;
	disablePan?: boolean;
	disableFocusCapture?: boolean;
	circularBehavior?: CircularBehavior;
	renderNodeHeader?: NodeHeaderRenderCallback;
	debug?: boolean;
	className?: string;
	edgeRoutingMode?: EdgeRoutingMode;
	graphLayoutMode?: GraphLayoutMode;
	style?: React.CSSProperties;
}

export const NodeEditor = ({
	ref,
	comments: initialComments,
	nodes: initialNodes,
	nodeTypes = {},
	portTypes = {},
	defaultNodes = [],
	defaultConnections = [],
	context = defaultContext,
	onChange,
	onCommentsChange,
	initialScale,
	spaceToPan = false,
	hideComments = false,
	disableComments = false,
	disableZoom = false,
	disablePan = false,
	disableFocusCapture = false,
	circularBehavior,
	renderNodeHeader,
	debug,
	className,
	style,
	edgeRoutingMode = "smooth",
	graphLayoutMode = "freeform",
}: NodeEditorProps) => {
	const editorId = useId() ?? "";
	const cache = React.useRef(new Cache());
	const stage = React.useRef<DOMRect | undefined>(undefined);
	const scaleRef = React.useRef(1);
	const environmentRef = React.useRef({
		nodeTypes,
		portTypes,
		cache,
		circularBehavior,
		context,
	});

	environmentRef.current = {
		nodeTypes,
		portTypes,
		cache,
		circularBehavior,
		context,
	};

	const [sideEffectToasts, setSideEffectToasts] = React.useState<ToastAction>();

	const [nodes, dispatchNodes] = React.useReducer(
		connectNodesReducer(
			nodesReducer,
			() => environmentRef.current,
			setSideEffectToasts,
		),
		{},
		() =>
			getInitialNodes(
				initialNodes,
				defaultNodes,
				nodeTypes,
				portTypes,
				context,
				defaultConnections,
			),
	);

	const [comments, dispatchComments] = React.useReducer(
		commentsReducer,
		initialComments || {},
	);

	const nodeTypeRegistryKey = React.useMemo(
		() => Object.keys(nodeTypes).sort().join("\0"),
		[nodeTypes],
	);
	const portTypeRegistryKey = React.useMemo(
		() => Object.keys(portTypes).sort().join("\0"),
		[portTypes],
	);

	// biome-ignore lint/correctness/useExhaustiveDependencies: reconcile graph when node/port registries change
	React.useEffect(() => {
		dispatchNodes({ type: NodesActionType.RECONCILE_NODE_TYPES });
	}, [nodeTypeRegistryKey, portTypeRegistryKey]);

	const visibleNodes = React.useMemo(
		() => Object.values(nodes).filter((node) => nodeTypes[node.type]),
		[nodes, nodeTypes],
	);

	const [stageState, dispatchStageState] = React.useReducer(stageReducer, {
		scale:
			typeof initialScale === "number"
				? Math.min(7, Math.max(0.1, initialScale))
				: 1,
		translate: { x: 0, y: 0 },
	});

	React.useLayoutEffect(() => {
		scaleRef.current = stageState.scale;
	}, [stageState.scale]);

	const [dragOverride, setDragOverride] = React.useState<Record<
		string,
		{ x: number; y: number }
	> | null>(null);

	const nodesRef = React.useRef(nodes);
	nodesRef.current = nodes;

	const onNodeLayoutChange = React.useCallback(() => {
		graphWorkerRef.current?.recalculate(nodesRef.current);
	}, []);

	const { indexRef, registerPortLayout: registerPortLayoutBase } =
		useSpatialIndex(editorId, dispatchNodes, onNodeLayoutChange);
	const graphWorkerBase = useFlumeGraphWorker(
		editorId,
		edgeRoutingMode,
		indexRef,
	);
	const graphWorkerRef = React.useRef(graphWorkerBase);
	graphWorkerRef.current = graphWorkerBase;

	const graphWorker = React.useMemo(
		() => ({
			beginDrag: (nodeId: string) => {
				graphWorkerBase.beginDrag(nodeId);
			},
			updateDrag: (nodeId: string, x: number, y: number) => {
				setDragOverride({ [nodeId]: { x, y } });
				graphWorkerBase.updateDrag(nodeId, x, y);
			},
			endDrag: (nodeId: string, x: number, y: number) => {
				setDragOverride(null);
				graphWorkerBase.endDrag(nodeId, x, y);
			},
			recalculate: graphWorkerBase.recalculate,
		}),
		[graphWorkerBase],
	);

	const recalculateConnections = React.useCallback(
		(positionOverrides?: Record<string, { x: number; y: number }>) => {
			createConnections(
				nodes,
				stageState,
				editorId,
				edgeRoutingMode,
				indexRef.current,
				positionOverrides,
			);
			graphWorker.recalculate(nodes, positionOverrides);
		},
		[nodes, editorId, stageState, edgeRoutingMode, indexRef, graphWorker],
	);

	const recalculateConnectionsRef = React.useRef(recalculateConnections);
	recalculateConnectionsRef.current = recalculateConnections;

	const portRecalcFrameRef = React.useRef<number | null>(null);

	const schedulePortLayoutRecalculate = React.useCallback(() => {
		if (portRecalcFrameRef.current !== null) {
			return;
		}

		portRecalcFrameRef.current = requestAnimationFrame(() => {
			portRecalcFrameRef.current = null;
			recalculateConnectionsRef.current();
		});
	}, []);

	const registerPortLayout = React.useCallback<
		NonNullable<React.ContextType<typeof PortLayoutRegistrationContext>>
	>(
		(nodeId, portName, transputType, entry) => {
			const layoutKey = portLayoutKey(nodeId, portName, transputType);
			const previous = indexRef.current.portLayouts.get(layoutKey);

			if (
				previous &&
				previous.offsetX === entry.offsetX &&
				previous.offsetY === entry.offsetY
			) {
				return;
			}

			registerPortLayoutBase(nodeId, portName, transputType, entry);
			schedulePortLayoutRecalculate();
		},
		[indexRef, registerPortLayoutBase, schedulePortLayoutRecalculate],
	);

	React.useEffect(() => {
		return () => {
			if (portRecalcFrameRef.current !== null) {
				cancelAnimationFrame(portRecalcFrameRef.current);
			}
		};
	}, []);

	const recalculateStageRect = React.useCallback(() => {
		stage.current = document
			.getElementById(`${STAGE_ID}${editorId}`)
			?.getBoundingClientRect();
	}, [editorId]);

	React.useLayoutEffect(() => {
		recalculateConnections();
	}, [recalculateConnections]);

	const triggerRecalculation = React.useCallback(
		(positionOverrides?: Record<string, { x: number; y: number }>) => {
			recalculateConnections(positionOverrides);
		},
		[recalculateConnections],
	);

	const nodesRefForLayout = nodesRef;

	const prevGraphLayout = usePrevious(graphLayoutMode);

	React.useEffect(() => {
		const mode = graphLayoutMode ?? "freeform";
		if (prevGraphLayout === undefined || prevGraphLayout === mode) return;
		if (mode === "freeform") return;
		dispatchGraphLayout(mode, nodesRefForLayout.current, dispatchNodes);
		triggerRecalculation();
	}, [
		graphLayoutMode,
		nodesRefForLayout,
		prevGraphLayout,
		triggerRecalculation,
	]);

	React.useImperativeHandle(ref, () => ({
		getNodes: () => {
			return nodes;
		},
		getComments: () => {
			return comments;
		},
	}));

	const previousNodes = usePrevious(nodes);

	React.useEffect(() => {
		if (previousNodes && onChange && nodes !== previousNodes) {
			onChange(nodes);
		}
	}, [nodes, previousNodes, onChange]);

	const previousComments = usePrevious(comments);

	React.useEffect(() => {
		if (previousComments && onCommentsChange && comments !== previousComments) {
			onCommentsChange(comments);
		}
	}, [comments, previousComments, onCommentsChange]);

	React.useEffect(() => {
		if (sideEffectToasts) {
			dispatchFlumeToastAction(sideEffectToasts);
			setSideEffectToasts(undefined);
		}
	}, [sideEffectToasts]);

	return (
		<Flex.Column
			className={cn("min-h-0 flex-1", className)}
			style={style}
			fullHeight
			fullWidth
		>
			<ObstacleIndexContext.Provider value={indexRef}>
				<PortLayoutRegistrationContext.Provider value={registerPortLayout}>
					<FlumeGraphWorkerContext.Provider value={graphWorker}>
						<NodeDragOverrideContext.Provider value={dragOverride}>
							<NodeMapContext.Provider value={nodes}>
								<EdgeRoutingContext.Provider value={edgeRoutingMode}>
									<PortTypesContext.Provider value={portTypes}>
										<NodeTypesContext.Provider value={nodeTypes}>
											<NodeDispatchContext.Provider value={dispatchNodes}>
												<ConnectionRecalculateContext.Provider
													value={triggerRecalculation}
												>
													<ContextContext.Provider value={context}>
														<StageContext.Provider value={stageState}>
															<CacheContext.Provider value={cache}>
																<EditorIdContext.Provider value={editorId}>
																	<RecalculateStageRectContext.Provider
																		value={recalculateStageRect}
																	>
																		<Stage
																			editorId={editorId}
																			scale={stageState.scale}
																			translate={stageState.translate}
																			spaceToPan={spaceToPan}
																			disablePan={disablePan}
																			disableZoom={disableZoom}
																			dispatchStageState={dispatchStageState}
																			dispatchComments={dispatchComments}
																			disableComments={
																				disableComments || hideComments
																			}
																			disableFocusCapture={disableFocusCapture}
																			stageRef={stage}
																			numNodes={Object.keys(nodes).length}
																			outerStageChildren={
																				debug ? (
																					<div className={styles.debugWrapper}>
																						<Button
																							type="button"
																							variant="outline"
																							size="sm"
																							onClick={() => console.log(nodes)}
																						>
																							Log Nodes
																						</Button>
																						<Button
																							type="button"
																							variant="outline"
																							size="sm"
																							onClick={() =>
																								console.log(
																									JSON.stringify(nodes),
																								)
																							}
																						>
																							Export Nodes
																						</Button>
																						<Button
																							type="button"
																							variant="outline"
																							size="sm"
																							onClick={() =>
																								console.log(comments)
																							}
																						>
																							Log Comments
																						</Button>
																					</div>
																				) : null
																			}
																		>
																			<div
																				className={styles.portLayer}
																				id={`${PORT_LAYER_ID}${editorId}`}
																			/>
																			{!hideComments &&
																				Object.values(comments).map(
																					(comment) => (
																						<Comment
																							{...comment}
																							stageRect={stage}
																							dispatch={dispatchComments}
																							onDragStart={recalculateStageRect}
																							key={comment.id}
																						/>
																					),
																				)}
																			{visibleNodes.map((node) => (
																				<Node
																					{...node}
																					stageRect={stage}
																					onDragStart={recalculateStageRect}
																					renderNodeHeader={renderNodeHeader}
																					key={node.id}
																				/>
																			))}
																			<Connections editorId={editorId} />
																			<div
																				className={styles.dragWrapper}
																				id={`${DRAG_CONNECTION_ID}${editorId}`}
																			/>
																		</Stage>
																	</RecalculateStageRectContext.Provider>
																</EditorIdContext.Provider>
															</CacheContext.Provider>
														</StageContext.Provider>
													</ContextContext.Provider>
												</ConnectionRecalculateContext.Provider>
											</NodeDispatchContext.Provider>
										</NodeTypesContext.Provider>
									</PortTypesContext.Provider>
								</EdgeRoutingContext.Provider>
							</NodeMapContext.Provider>
						</NodeDragOverrideContext.Provider>
					</FlumeGraphWorkerContext.Provider>
				</PortLayoutRegistrationContext.Provider>
			</ObstacleIndexContext.Provider>
		</Flex.Column>
	);
};
