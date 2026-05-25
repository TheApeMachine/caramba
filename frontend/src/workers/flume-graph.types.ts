import type { EdgeRoutingMode } from "#/components/flume/connectionCalculator";
import type {
	Coordinate,
	NodeMap,
	TransputType,
} from "#/components/flume/types";

export type ConnectionPathResult = {
	id: string;
	d: string;
};

export type GraphPortLayout = {
	nodeId: string;
	portName: string;
	transputType: TransputType;
	offsetX: number;
	offsetY: number;
};

export type GraphNodeLayout = {
	nodeId: string;
	width: number;
	height: number;
};

export type GraphSnapshot = {
	nodes: NodeMap;
	routingMode: EdgeRoutingMode;
	portLayouts: GraphPortLayout[];
	nodeLayouts: GraphNodeLayout[];
};

export type DragUpdateResult = {
	paths: ConnectionPathResult[];
	dragPosition: Coordinate;
};
