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
	ObstacleIndexContext,
	useConnectionWorker,
	useObstacleIndex,
} from "#/components/flume/useObstacleIndex";
import { DRAG_CONNECTION_ID, STAGE_ID } from "#/components/flume/constants";
import Node from "#/components/flume/Node/Node";
import Stage from "#/components/flume/Stage/Stage";
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
	NodeDispatchContext,
	NodeMapContext,
	NodeTypesContext,
	PortTypesContext,
	RecalculateConnectionsWorkerContext,
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
	DefaultNode,
	FlumeCommentMap,
	NodeHeaderRenderCallback,
	NodeMap,
	NodeTypeMap,
	PortTypeMap,
} from "./types";

const defaultContext = {};

interface NodeEditorProps {
	comments?: FlumeCommentMap;
	nodes?: NodeMap;
	nodeTypes: NodeTypeMap;
	portTypes: PortTypeMap;
	defaultNodes?: DefaultNode[];
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

export const NodeEditor = React.forwardRef(
	(
		{
			comments: initialComments,
			nodes: initialNodes,
			nodeTypes = {},
			portTypes = {},
			defaultNodes = [],
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
		}: NodeEditorProps,
		ref,
	) => {
		const editorId = useId() ?? "";
		const cache = React.useRef(new Cache());
		const stage = React.useRef<DOMRect | undefined>(undefined);
		const scaleRef = React.useRef(1);

		const [sideEffectToasts, setSideEffectToasts] =
			React.useState<ToastAction>();

		const [nodes, dispatchNodes] = React.useReducer(
			connectNodesReducer(
				nodesReducer,
				{ nodeTypes, portTypes, cache, circularBehavior, context },
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
				),
		);

		const [comments, dispatchComments] = React.useReducer(
			commentsReducer,
			initialComments || {},
		);

		React.useEffect(() => {
			dispatchNodes({ type: NodesActionType.HYDRATE_DEFAULT_NODES });
		}, []);

		const [shouldRecalculateConnections, setShouldRecalculateConnections] =
			React.useState(true);

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

		const obstacleIndex = useObstacleIndex(editorId, stageState.scale);
		const recalculateWorker = useConnectionWorker(
			editorId,
			edgeRoutingMode,
			obstacleIndex,
			scaleRef,
		);

		const recalculateConnections = React.useCallback(() => {
			createConnections(nodes, stageState, editorId, edgeRoutingMode);
		}, [nodes, editorId, stageState, edgeRoutingMode]);

		const recalculateStageRect = React.useCallback(() => {
			stage.current = document
				.getElementById(`${STAGE_ID}${editorId}`)
				?.getBoundingClientRect();
		}, [editorId]);

		React.useLayoutEffect(() => {
			if (shouldRecalculateConnections) {
				recalculateConnections();
				setShouldRecalculateConnections(false);
			}
		}, [shouldRecalculateConnections, recalculateConnections]);

		const triggerRecalculation = React.useCallback(() => {
			setShouldRecalculateConnections(true);
		}, []);

		const nodesRef = React.useRef(nodes);
		nodesRef.current = nodes;

		const prevGraphLayout = usePrevious(graphLayoutMode);

		React.useEffect(() => {
			const mode = graphLayoutMode ?? "freeform";
			if (prevGraphLayout === undefined || prevGraphLayout === mode) return;
			if (mode === "freeform") return;
			dispatchGraphLayout(mode, nodesRef.current, dispatchNodes);
			triggerRecalculation();
		}, [graphLayoutMode, prevGraphLayout, triggerRecalculation]);

		React.useEffect(() => {
			void edgeRoutingMode;
			setShouldRecalculateConnections(true);
		}, [edgeRoutingMode]);

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
			if (
				previousComments &&
				onCommentsChange &&
				comments !== previousComments
			) {
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
				<ObstacleIndexContext.Provider value={obstacleIndex}>
					<RecalculateConnectionsWorkerContext.Provider value={recalculateWorker}>
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
																		disableComments={disableComments || hideComments}
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
																							console.log(JSON.stringify(nodes))
																						}
																					>
																						Export Nodes
																					</Button>
																					<Button
																						type="button"
																						variant="outline"
																						size="sm"
																						onClick={() => console.log(comments)}
																					>
																						Log Comments
																					</Button>
																				</div>
																			) : null
																		}
																	>
																		{!hideComments &&
																			Object.values(comments).map((comment) => (
																				<Comment
																					{...comment}
																					stageRect={stage}
																					dispatch={dispatchComments}
																					onDragStart={recalculateStageRect}
																					key={comment.id}
																				/>
																			))}
																		{Object.values(nodes).map((node) => (
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
					</RecalculateConnectionsWorkerContext.Provider>
				</ObstacleIndexContext.Provider>
			</Flex.Column>
		);
	},
);
