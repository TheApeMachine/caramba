import React from "react";
import {
	calculateEdgePath,
	type EdgeRoutingMode,
	getPortRect,
	getStageBounds,
	readLiveStageScale,
	screenPointToCanvas,
} from "#/components/flume/connectionCalculator";
import {
	EditorIdContext,
	NodeDispatchContext,
	PortTypesContext,
	StageContext,
	useEdgeRouting,
} from "#/components/flume/context";
import type { NodesAction } from "#/components/flume/nodesReducer";
import { NodesActionType } from "#/components/flume/nodesReducer";
import type { Coordinate, PortTypeMap } from "#/components/flume/types";

type NodesDispatch = React.Dispatch<NodesAction> | null;

/*
usePortDrag isolates the drag-line state machine that used to live inside
the Port component. The Port component now only owns the visual anchor
and portal, while this hook owns:
  - mouse event registration / cleanup
  - dragStartCoordinates state
  - main-thread SVG `d` attribute updates against the routing mode
  - drop resolution (ADD_CONNECTION / REMOVE_CONNECTION dispatch)
*/

interface PortDragOptions {
	nodeId: string;
	name: string;
	type: string;
	isInput?: boolean;
	triggerRecalculation: () => void;
	portButtonRef: React.RefObject<HTMLButtonElement | null>;
}

export interface PortDragHandle {
	isDragging: boolean;
	dragStartCoordinates: Coordinate;
	lineRef: React.RefObject<SVGPathElement | null>;
	handleDragStart: (e: React.MouseEvent<HTMLButtonElement>) => void;
	beginDragFromPort: () => void;
}

const clientPointToCanvasAt = (
	editorId: string,
	fallbackScale: number,
	clientX: number,
	clientY: number,
): Coordinate => {
	const stageRect = getStageBounds(editorId);

	if (!stageRect) {
		return { x: 0, y: 0 };
	}

	const scale = readLiveStageScale(editorId, fallbackScale);
	return screenPointToCanvas(clientX, clientY, stageRect, scale);
};

const portRectCenterToCanvas = (
	editorId: string,
	fallbackScale: number,
	rect: DOMRect | null | undefined,
): Coordinate => {
	if (!rect) {
		return { x: 0, y: 0 };
	}

	return clientPointToCanvasAt(
		editorId,
		fallbackScale,
		rect.left + rect.width / 2,
		rect.top + rect.height / 2,
	);
};

const dispatchAcceptedConnection = ({
	target,
	type,
	nodeId,
	name,
	inputTypes,
	nodesDispatch,
	triggerRecalculation,
}: {
	target: HTMLElement;
	type: string;
	nodeId: string;
	name: string;
	inputTypes: PortTypeMap;
	nodesDispatch: NodesDispatch;
	triggerRecalculation: () => void;
}) => {
	const {
		portName: inputPortName,
		nodeId: inputNodeId,
		portType: inputNodeType,
		portTransputType: inputTransputType,
	} = target.dataset;

	if (
		!inputPortName ||
		!inputNodeId ||
		!inputNodeType ||
		!inputTransputType ||
		inputNodeId === nodeId ||
		inputTransputType === "output"
	) {
		return;
	}

	const willAccept =
		inputTypes?.[inputNodeType]?.acceptTypes?.includes(type) ?? false;

	if (!willAccept) {
		return;
	}

	nodesDispatch?.({
		type: NodesActionType.ADD_CONNECTION,
		output: { nodeId, portName: name },
		input: { nodeId: inputNodeId, portName: inputPortName },
	});
	triggerRecalculation();
};

const resolveInputDrop = ({
	target,
	outputNodeId,
	outputPortName,
	type,
	inputTypes,
	nodesDispatch,
}: {
	target: HTMLElement;
	outputNodeId: string;
	outputPortName: string;
	type: string;
	inputTypes: PortTypeMap;
	nodesDispatch: NodesDispatch;
}) => {
	const {
		portName: connectToPortName,
		nodeId: connectToNodeId,
		portType: connectToPortType,
		portTransputType: connectToTransputType,
	} = target.dataset;

	if (
		!connectToPortName ||
		!connectToNodeId ||
		!connectToPortType ||
		!connectToTransputType ||
		outputNodeId === connectToNodeId ||
		connectToTransputType === "output"
	) {
		return;
	}

	const willAccept =
		inputTypes?.[connectToPortType]?.acceptTypes?.includes(type) ?? false;

	if (!willAccept) {
		return;
	}

	nodesDispatch?.({
		type: NodesActionType.ADD_CONNECTION,
		input: { nodeId: connectToNodeId, portName: connectToPortName },
		output: { nodeId: outputNodeId, portName: outputPortName },
	});
};

export const usePortDrag = ({
	nodeId,
	name,
	type,
	isInput,
	triggerRecalculation,
	portButtonRef,
}: PortDragOptions): PortDragHandle => {
	const nodesDispatch = React.useContext(NodeDispatchContext);
	const stageState = React.useContext(StageContext) || {
		scale: 1,
		translate: { x: 0, y: 0 },
	};
	const editorId = React.useContext(EditorIdContext);
	const inputTypes = React.useContext(PortTypesContext) ?? {};
	const edgeRouting: EdgeRoutingMode = useEdgeRouting();

	const [isDragging, setIsDragging] = React.useState(false);
	const [dragStartCoordinates, setDragStartCoordinates] =
		React.useState<Coordinate>({ x: 0, y: 0 });
	const dragStartCoordinatesCache = React.useRef(dragStartCoordinates);
	const line = React.useRef<SVGPathElement | null>(null);
	const lineInToPort = React.useRef<SVGPathElement | null>(null);

	const handleDrag = React.useCallback(
		(event: MouseEvent) => {
			const to = clientPointToCanvasAt(
				editorId,
				stageState.scale ?? 1,
				event.clientX,
				event.clientY,
			);
			const d = calculateEdgePath(
				edgeRouting,
				dragStartCoordinatesCache.current,
				to,
			);

			if (isInput) {
				lineInToPort.current?.setAttribute("d", d);
				return;
			}

			line.current?.setAttribute("d", d);
		},
		[editorId, stageState.scale, edgeRouting, isInput],
	);

	const handleDragEnd = React.useCallback(
		(e: MouseEvent) => {
			const target = e.target as HTMLElement;
			const droppedOnPort = !!target?.dataset?.portName;

			if (isInput) {
				const {
					inputNodeId = "",
					inputPortName = "",
					outputNodeId = "",
					outputPortName = "",
				} = lineInToPort.current?.dataset ?? {};

				nodesDispatch?.({
					type: NodesActionType.REMOVE_CONNECTION,
					input: { nodeId: inputNodeId, portName: inputPortName },
					output: { nodeId: outputNodeId, portName: outputPortName },
				});

				if (droppedOnPort) {
					resolveInputDrop({
						target,
						outputNodeId,
						outputPortName,
						type,
						inputTypes,
						nodesDispatch,
					});
				}
			} else if (droppedOnPort) {
				dispatchAcceptedConnection({
					target,
					type,
					nodeId,
					name,
					inputTypes,
					nodesDispatch,
					triggerRecalculation,
				});
			}

			setIsDragging(false);
			document.removeEventListener("mouseup", handleDragEnd);
			document.removeEventListener("mousemove", handleDrag);
		},
		[
			handleDrag,
			isInput,
			nodesDispatch,
			type,
			inputTypes,
			nodeId,
			name,
			triggerRecalculation,
		],
	);

	const beginDragFromPort = React.useCallback(() => {
		if (isInput) {
			lineInToPort.current = document.querySelector<SVGPathElement>(
				`[data-input-node-id="${nodeId}"][data-input-port-name="${name}"]`,
			);

			const portIsConnected = !!lineInToPort.current;

			if (
				!portIsConnected ||
				!lineInToPort.current ||
				!lineInToPort.current.parentElement
			) {
				return;
			}

			lineInToPort.current.parentElement.style.zIndex = "9999";

			const outputRect = getPortRect(
				lineInToPort.current.dataset.outputNodeId || "",
				lineInToPort.current.dataset.outputPortName || "",
				"output",
			);
			const coordinates = portRectCenterToCanvas(
				editorId,
				stageState.scale ?? 1,
				outputRect,
			);
			setDragStartCoordinates(coordinates);
			dragStartCoordinatesCache.current = coordinates;
			setIsDragging(true);
			document.addEventListener("mouseup", handleDragEnd);
			document.addEventListener("mousemove", handleDrag);
			return;
		}

		const coordinates = portRectCenterToCanvas(
			editorId,
			stageState.scale ?? 1,
			portButtonRef.current?.getBoundingClientRect(),
		);
		setDragStartCoordinates(coordinates);
		dragStartCoordinatesCache.current = coordinates;
		setIsDragging(true);
		document.addEventListener("mouseup", handleDragEnd);
		document.addEventListener("mousemove", handleDrag);
	}, [
		isInput,
		nodeId,
		name,
		editorId,
		stageState.scale,
		handleDrag,
		handleDragEnd,
		portButtonRef,
	]);

	const handleDragStart = React.useCallback(
		(e: React.MouseEvent<HTMLButtonElement>) => {
			e.preventDefault();
			e.stopPropagation();
			beginDragFromPort();
		},
		[beginDragFromPort],
	);

	return {
		isDragging,
		dragStartCoordinates,
		lineRef: line,
		handleDragStart,
		beginDragFromPort,
	};
};
