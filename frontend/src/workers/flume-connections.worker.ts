import {
	calculateEdgePath,
	type EdgeRoutingMode,
	type ObstacleRect,
} from "#/components/flume/connectionCalculator";
import type { Coordinate } from "#/components/flume/types";

export type ConnectionPathRequest = {
	id: string;
	from: Coordinate;
	to: Coordinate;
	routingMode: EdgeRoutingMode;
	obstaclesVertical: ObstacleRect[];
	obstaclesHorizontal: ObstacleRect[];
};

export type ConnectionPathResult = {
	id: string;
	d: string;
};

export type WorkerRequest = {
	connections: ConnectionPathRequest[];
};

export type WorkerResponse = {
	paths: ConnectionPathResult[];
};

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
	const paths: ConnectionPathResult[] = event.data.connections.map(
		({ id, from, to, routingMode, obstaclesVertical, obstaclesHorizontal }) => ({
			id,
			d: calculateEdgePath(routingMode, from, to, obstaclesVertical, obstaclesHorizontal),
		}),
	);

	self.postMessage({ paths } satisfies WorkerResponse);
};
