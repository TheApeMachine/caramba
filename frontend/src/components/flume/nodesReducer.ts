import type { RefObject } from "react";
import type FlumeCache from "./Cache";
import {
	deleteConnection,
	deleteConnectionsByNodeId,
} from "./connectionCalculator";
import {
	getDefaultData,
	pruneDanglingConnections,
	reconcileNodes,
} from "./nodes-helpers";
import type { ToastAction } from "./toastsReducer";
import type {
	CircularBehavior,
	Connection,
	ConnectionMap,
	Connections,
	DefaultConnection,
	DefaultNode,
	FlumeNode,
	InputData,
	NodeMap,
	NodeTypeMap,
	PortTypeMap,
	TransputType,
	ValueSetter,
} from "./types";
import { checkForCircularNodes, createFlumeId } from "./utilities";

/*
Re-export the pure topology helpers so existing consumers that import
from "./nodesReducer" keep working. The implementations live in
nodes-helpers; this barrel exists for backward compatibility.
*/
export {
	getDefaultData,
	normalizeFlumeNode,
	pruneDanglingConnections,
	reconcileNodes,
} from "./nodes-helpers";

export enum NodesActionType {
	ADD_CONNECTION = "ADD_CONNECTION",
	REMOVE_CONNECTION = "REMOVE_CONNECTION",
	DESTROY_TRANSPUT = "DESTROY_TRANSPUT",
	ADD_NODE = "ADD_NODE",
	REMOVE_NODE = "REMOVE_NODE",
	HYDRATE_DEFAULT_NODES = "HYDRATE_DEFAULT_NODES",
	SET_PORT_DATA = "SET_PORT_DATA",
	SET_NODE_COORDINATES = "SET_NODE_COORDINATES",
	SET_NODE_DIMENSIONS = "SET_NODE_DIMENSIONS",
	SET_NODE_SUBGRAPH = "SET_NODE_SUBGRAPH",
	RECONCILE_NODE_TYPES = "RECONCILE_NODE_TYPES",
}

type ProposedConnection = { nodeId: string; portName: string };

const addConnection = (
	nodes: NodeMap,
	input: ProposedConnection,
	output: ProposedConnection,
) => {
	const inputNode = nodes[input.nodeId];
	const outputNode = nodes[output.nodeId];

	if (!inputNode?.connections || !outputNode?.connections) {
		return nodes;
	}

	const newNodes = {
		...nodes,
		[input.nodeId]: {
			...inputNode,
			connections: {
				...inputNode.connections,
				inputs: {
					...inputNode.connections.inputs,
					[input.portName]: [
						...(inputNode.connections.inputs[input.portName] || []),
						{
							nodeId: output.nodeId,
							portName: output.portName,
						},
					],
				},
			},
		},
		[output.nodeId]: {
			...outputNode,
			connections: {
				...outputNode.connections,
				outputs: {
					...outputNode.connections.outputs,
					[output.portName]: [
						...(outputNode.connections.outputs[output.portName] || []),
						{
							nodeId: input.nodeId,
							portName: input.portName,
						},
					],
				},
			},
		},
	};
	return newNodes;
};

const removeConnection = (
	nodes: NodeMap,
	input: ProposedConnection,
	output: ProposedConnection,
) => {
	const inputNode = nodes[input.nodeId];
	const {
		[input.portName]: _removedInputPort,
		...newInputNodeConnectionsInputs
	} = inputNode.connections.inputs;
	const newInputNode = {
		...inputNode,
		connections: {
			...inputNode.connections,
			inputs: newInputNodeConnectionsInputs,
		},
	};

	const outputNode = nodes[output.nodeId];
	const filteredOutputNodes = outputNode.connections.outputs[
		output.portName
	].filter((cnx) => {
		return cnx.nodeId === input.nodeId ? cnx.portName !== input.portName : true;
	});
	const newOutputNode = {
		...outputNode,
		connections: {
			...outputNode.connections,
			outputs: {
				...outputNode.connections.outputs,
				[output.portName]: filteredOutputNodes,
			},
		},
	};

	return {
		...nodes,
		[input.nodeId]: newInputNode,
		[output.nodeId]: newOutputNode,
	};
};

const getFilteredTransputs = (transputs: ConnectionMap, nodeId: string) =>
	Object.entries(transputs).reduce<{ [key: string]: Connection[] }>(
		(obj, [portName, transput]) => {
			const newTransputs = transput.filter((t) => t.nodeId !== nodeId);
			if (newTransputs.length) {
				obj[portName] = newTransputs;
			}
			return obj;
		},
		{},
	);

const removeConnections = (connections: Connections, nodeId: string) => ({
	inputs: getFilteredTransputs(connections.inputs, nodeId),
	outputs: getFilteredTransputs(connections.outputs, nodeId),
});

const remapConnectionNodeIds = (
	connections: Connections,
	idMap: Map<string, string>,
): Connections => {
	const remapLinks = (links: Connection[]) =>
		links.map((link) => ({
			...link,
			nodeId: idMap.get(link.nodeId) ?? link.nodeId,
		}));

	return {
		inputs: Object.fromEntries(
			Object.entries(connections.inputs).map(([portName, links]) => [
				portName,
				remapLinks(links),
			]),
		),
		outputs: Object.fromEntries(
			Object.entries(connections.outputs).map(([portName, links]) => [
				portName,
				remapLinks(links),
			]),
		),
	};
};

const removeNode = (startNodes: NodeMap, nodeId: string) => {
	let { [nodeId]: _deletedNode, ...nodes } = startNodes;
	nodes = Object.values(nodes).reduce<NodeMap>((obj, node) => {
		obj[node.id] = {
			...node,
			connections: removeConnections(node.connections, nodeId),
		};

		return obj;
	}, {});
	deleteConnectionsByNodeId(nodeId);
	return nodes;
};

const findNodeByType = (
	nodes: NodeMap,
	nodeType: string,
): FlumeNode | undefined =>
	Object.values(nodes).find((node) => node.type === nodeType);

/*
applyDefaultConnections wires demo edges by node type after default nodes exist.
Runs during init so connection endpoints use stable node ids from the start.
*/
export const applyDefaultConnections = (
	nodes: NodeMap,
	defaultConnections: DefaultConnection[],
): NodeMap => {
	let nextNodes = nodes;

	for (const connection of defaultConnections) {
		const outputNode = findNodeByType(nextNodes, connection.output.nodeType);
		const inputNode = findNodeByType(nextNodes, connection.input.nodeType);

		if (!outputNode || !inputNode) {
			continue;
		}

		const existingInputs =
			inputNode.connections?.inputs[connection.input.portName] ?? [];
		const hasValidConnection = existingInputs.some(
			(link) =>
				link.nodeId === outputNode.id &&
				link.portName === connection.output.portName,
		);

		if (hasValidConnection) {
			continue;
		}

		nextNodes = addConnection(
			nextNodes,
			{ nodeId: inputNode.id, portName: connection.input.portName },
			{ nodeId: outputNode.id, portName: connection.output.portName },
		);
	}

	return nextNodes;
};

export const getInitialNodes = (
	initialNodes: NodeMap = {},
	defaultNodes: DefaultNode[] = [],
	nodeTypes: NodeTypeMap,
	portTypes: PortTypeMap,
	context: unknown,
	defaultConnections: DefaultConnection[] = [],
): NodeMap => {
	const reconciledNodes = reconcileNodes(
		initialNodes,
		nodeTypes,
		portTypes,
		context,
	);

	const withDefaultNodes = defaultNodes.reduce((nodes, dNode) => {
		const nodeNotAdded = !Object.values(initialNodes).find(
			(node) => node.type === dNode.type,
		);

		if (nodeNotAdded && nodeTypes[dNode.type]) {
			return nodesReducer(
				nodes,
				{
					type: NodesActionType.ADD_NODE,
					id: createFlumeId(),
					x: dNode.x || 0,
					y: dNode.y || 0,
					nodeType: dNode.type,
				},
				{ nodeTypes, portTypes, context },
			);
		}

		return nodes;
	}, reconciledNodes);

	return applyDefaultConnections(withDefaultNodes, defaultConnections);
};

export type NodesAction =
	| {
			type: NodesActionType.ADD_CONNECTION | NodesActionType.REMOVE_CONNECTION;
			input: ProposedConnection;
			output: ProposedConnection;
	  }
	| {
			type: NodesActionType.DESTROY_TRANSPUT;
			transput: ProposedConnection;
			transputType: TransputType;
	  }
	| {
			type: NodesActionType.ADD_NODE;
			nodeType: string;
			x: number;
			y: number;
			id?: string;
			defaultNode?: boolean;
	  }
	| {
			type: NodesActionType.REMOVE_NODE;
			nodeId: string;
	  }
	| {
			type: NodesActionType.HYDRATE_DEFAULT_NODES;
	  }
	| {
			type: NodesActionType.SET_PORT_DATA;
			nodeId: string;
			portName: string;
			controlName: string;
			data: unknown;
			setValue?: ValueSetter;
	  }
	| {
			type: NodesActionType.SET_NODE_COORDINATES;
			x: number;
			y: number;
			nodeId: string;
	  }
	| {
			type: NodesActionType.SET_NODE_DIMENSIONS;
			nodeId: string;
			width: number;
			height: number;
	  }
	| {
			type: NodesActionType.SET_NODE_SUBGRAPH;
			nodeId: string;
			subGraph: NodeMap;
	  }
	| { type: NodesActionType.RECONCILE_NODE_TYPES };

interface FlumeEnvironment {
	nodeTypes: NodeTypeMap;
	portTypes: PortTypeMap;
	cache?: RefObject<FlumeCache>;
	circularBehavior?: CircularBehavior;
	context: unknown;
}

const nodesReducer = (
	nodes: NodeMap,
	action: NodesAction,
	{ nodeTypes, portTypes, cache, circularBehavior, context }: FlumeEnvironment,
	_dispatchToasts?: React.Dispatch<
		React.SetStateAction<ToastAction | undefined>
	>,
) => {
	switch (action.type) {
		case NodesActionType.ADD_CONNECTION: {
			const { input, output } = action;
			const inputNode = nodes[input.nodeId];

			if (!inputNode?.connections) {
				return nodes;
			}

			const inputIsNotConnected = !inputNode.connections.inputs[input.portName];
			if (inputIsNotConnected) {
				const allowCircular =
					circularBehavior === "warn" || circularBehavior === "allow";
				const newNodes = addConnection(nodes, input, output);
				const isCircular = checkForCircularNodes(newNodes, output.nodeId);
				if (isCircular && !allowCircular) {
					return nodes;
				} else {
					if (isCircular && circularBehavior === "warn") {
						// warn-only: no-op in core library
					}
					return newNodes;
				}
			} else return nodes;
		}

		case NodesActionType.REMOVE_CONNECTION: {
			const { input, output } = action;
			const id = `${output.nodeId}|${output.portName}|${input.nodeId}|${input.portName}`;
			if (cache?.current?.connections) {
				delete cache.current.connections[id];
			}
			deleteConnection({ id });
			return removeConnection(nodes, input, output);
		}

		case NodesActionType.DESTROY_TRANSPUT: {
			const { transput, transputType } = action;
			const portId = transput.nodeId + transput.portName + transputType;
			if (cache?.current?.ports) {
				delete cache.current.ports[portId];
			}

			const cnxType = transputType === "input" ? "inputs" : "outputs";
			const connections =
				nodes[transput.nodeId].connections[cnxType][transput.portName];
			if (!connections || !connections.length) return nodes;

			return connections.reduce((nodes, cnx) => {
				const [input, output] =
					transputType === "input" ? [transput, cnx] : [cnx, transput];
				const id = `${output.nodeId}|${output.portName}|${input.nodeId}|${input.portName}`;
				if (cache?.current?.connections) {
					delete cache.current.connections[id];
				}
				deleteConnection({ id });
				return removeConnection(nodes, input, output);
			}, nodes);
		}

		case NodesActionType.ADD_NODE: {
			const { x, y, nodeType, id: _id, defaultNode: _defaultNode } = action;

			if (!nodeTypes[nodeType]) {
				return nodes;
			}

			const newNodeId = _id ?? createFlumeId();
			const newNode: FlumeNode = {
				id: newNodeId,
				x,
				y,
				type: nodeType,
				width: nodeTypes[nodeType].initialWidth ?? 200,
				connections: {
					inputs: {},
					outputs: {},
				},
				inputData: {},
			};
			newNode.inputData = getDefaultData({
				node: newNode,
				nodeType: nodeTypes[nodeType],
				portTypes,
				context,
			});
			if (_defaultNode) {
				newNode.defaultNode = true;
			}
			if (nodeTypes[nodeType].root) {
				newNode.root = true;
			}
			if (nodeTypes[nodeType].defaultSubGraph) {
				newNode.subGraph = nodeTypes[nodeType].defaultSubGraph;
			}
			return {
				...nodes,
				[newNodeId]: newNode,
			};
		}

		case NodesActionType.REMOVE_NODE: {
			const { nodeId } = action;
			return removeNode(nodes, nodeId);
		}

		case NodesActionType.HYDRATE_DEFAULT_NODES: {
			const idMap = new Map<string, string>();
			let newNodes = { ...nodes };

			for (const key of Object.keys(newNodes)) {
				if (!newNodes[key].defaultNode) {
					continue;
				}

				const newNodeId = createFlumeId();
				idMap.set(key, newNodeId);

				const { id: _oldId, defaultNode: _oldDefault, ...node } = newNodes[key];
				newNodes[newNodeId] = { ...node, id: newNodeId };
				delete newNodes[key];
			}

			if (idMap.size === 0) {
				return nodes;
			}

			newNodes = Object.fromEntries(
				Object.entries(newNodes).map(([nodeId, node]) => [
					nodeId,
					{
						...node,
						connections: remapConnectionNodeIds(node.connections, idMap),
					},
				]),
			);

			return pruneDanglingConnections(newNodes);
		}

		case NodesActionType.SET_PORT_DATA: {
			const { nodeId, portName, controlName, data, setValue } = action;
			let newData: Record<string, unknown> = {
				...nodes[nodeId].inputData,
				[portName]: {
					...nodes[nodeId].inputData[portName],
					[controlName]: data,
				},
			};
			if (setValue) {
				newData = setValue(newData, nodes[nodeId].inputData) as Record<
					string,
					unknown
				>;
			}
			return {
				...nodes,
				[nodeId]: {
					...nodes[nodeId],
					inputData: newData as InputData,
				},
			};
		}

		case NodesActionType.SET_NODE_COORDINATES: {
			const { x, y, nodeId } = action;
			return {
				...nodes,
				[nodeId]: {
					...nodes[nodeId],
					x,
					y,
				},
			};
		}

		case NodesActionType.SET_NODE_DIMENSIONS: {
			const { nodeId, width, height } = action;
			return {
				...nodes,
				[nodeId]: {
					...nodes[nodeId],
					width,
					height,
				},
			};
		}

		case NodesActionType.SET_NODE_SUBGRAPH: {
			const { nodeId, subGraph } = action;
			return {
				...nodes,
				[nodeId]: {
					...nodes[nodeId],
					subGraph,
				},
			};
		}

		case NodesActionType.RECONCILE_NODE_TYPES:
			return reconcileNodes(nodes, nodeTypes, portTypes, context);

		default:
			return nodes;
	}
};

export const connectNodesReducer =
	(
		reducer: typeof nodesReducer,
		readEnvironment: () => FlumeEnvironment,
		dispatchToasts: React.Dispatch<
			React.SetStateAction<ToastAction | undefined>
		>,
	) =>
	(state: NodeMap, action: NodesAction) =>
		reducer(state, action, readEnvironment(), dispatchToasts);

export default nodesReducer;
