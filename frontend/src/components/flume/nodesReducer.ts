import type { RefObject } from "react";
import type FlumeCache from "./Cache";
import {
	deleteConnection,
	deleteConnectionsByNodeId,
} from "./connectionCalculator";
import type { ToastAction } from "./toastsReducer";
import type {
	CircularBehavior,
	Connection,
	ConnectionMap,
	Connections,
	ControlData,
	DefaultNode,
	FlumeNode,
	InputData,
	NodeMap,
	NodeType,
	NodeTypeMap,
	PortTypeMap,
	TransputType,
	ValueSetter,
} from "./types";
import { checkForCircularNodes, createFlumeId } from "./utilities";

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

const addConnection = (
	nodes: NodeMap,
	input: ProposedConnection,
	output: ProposedConnection,
) => {
	const newNodes = {
		...nodes,
		[input.nodeId]: {
			...nodes[input.nodeId],
			connections: {
				...nodes[input.nodeId].connections,
				inputs: {
					...nodes[input.nodeId].connections.inputs,
					[input.portName]: [
						...(nodes[input.nodeId].connections.inputs[input.portName] || []),
						{
							nodeId: output.nodeId,
							portName: output.portName,
						},
					],
				},
			},
		},
		[output.nodeId]: {
			...nodes[output.nodeId],
			connections: {
				...nodes[output.nodeId].connections,
				outputs: {
					...nodes[output.nodeId].connections.outputs,
					[output.portName]: [
						...(nodes[output.nodeId].connections.outputs[output.portName] ||
							[]),
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

/*
pruneDanglingConnections removes input/output links whose endpoint node no
longer exists. Persisted graphs often keep stale ids after default-node hydration.
*/
export const pruneDanglingConnections = (nodes: NodeMap): NodeMap => {
	const nodeIds = new Set(Object.keys(nodes));
	let changed = false;
	const next: NodeMap = {};

	for (const [nodeId, node] of Object.entries(nodes)) {
		const inputs: ConnectionMap = {};

		for (const [portName, links] of Object.entries(node.connections.inputs)) {
			const filtered = links.filter((link) => nodeIds.has(link.nodeId));

			if (filtered.length !== links.length) {
				changed = true;
			}

			if (filtered.length > 0) {
				inputs[portName] = filtered;
			}
		}

		const outputs: ConnectionMap = {};

		for (const [portName, links] of Object.entries(node.connections.outputs)) {
			const filtered = links.filter((link) => nodeIds.has(link.nodeId));

			if (filtered.length !== links.length) {
				changed = true;
			}

			if (filtered.length > 0) {
				outputs[portName] = filtered;
			}
		}

		next[nodeId] = {
			...node,
			connections: { inputs, outputs },
		};
	}

	return changed ? next : nodes;
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

const reconcileNodes = (
	initialNodes: NodeMap,
	nodeTypes: NodeTypeMap,
	portTypes: PortTypeMap,
	context: unknown,
): NodeMap => {
	const knownNodes: NodeMap = {};

	for (const [nodeId, node] of Object.entries(initialNodes)) {
		if (node?.type && nodeTypes[node.type]) {
			knownNodes[nodeId] = node;
			continue;
		}

		if (typeof document !== "undefined") {
			deleteConnectionsByNodeId(nodeId);
		}
	}

	// Reconcile input data for each node
	let reconciledNodes = Object.values(knownNodes).reduce<NodeMap>(
		(nodesObj, node) => {
			const nodeType = nodeTypes[node.type];

			if (!nodeType) {
				return nodesObj;
			}

			const defaultInputData = getDefaultData({
				node,
				nodeType,
				portTypes,
				context,
			});
			const currentInputData = Object.entries(node.inputData).reduce<InputData>(
				(dataObj, [key, data]) => {
					if (defaultInputData[key] !== undefined) {
						dataObj[key] = data;
					}
					return dataObj;
				},
				{},
			);
			const newInputData = {
				...defaultInputData,
				...currentInputData,
			};
			nodesObj[node.id] = {
				...node,
				inputData: newInputData,
			};
			return nodesObj;
		},
		{},
	);

	// Reconcile node attributes for each node
	reconciledNodes = Object.values(reconciledNodes).reduce<NodeMap>(
		(nodesObj, node) => {
			const nodeType = nodeTypes[node.type];

			if (!nodeType) {
				return nodesObj;
			}

			const newNode = { ...node };
			if (nodeType.root !== node.root) {
				if (nodeType.root && !node.root) {
					newNode.root = nodeType.root;
				} else if (!nodeType.root && node.root) {
					delete newNode.root;
				}
			}
			nodesObj[node.id] = newNode;
			return nodesObj;
		},
		{},
	);

	return pruneDanglingConnections(reconciledNodes);
};

export const getInitialNodes = (
	initialNodes: NodeMap = {},
	defaultNodes: DefaultNode[] = [],
	nodeTypes: NodeTypeMap,
	portTypes: PortTypeMap,
	context: unknown,
): NodeMap => {
	const reconciledNodes = reconcileNodes(
		initialNodes,
		nodeTypes,
		portTypes,
		context,
	);

	return {
		...reconciledNodes,
		...defaultNodes.reduce((nodes, dNode, i) => {
			const nodeNotAdded = !Object.values(initialNodes).find(
				(n) => n.type === dNode.type,
			);
			if (nodeNotAdded && nodeTypes[dNode.type]) {
				nodes = nodesReducer(
					nodes,
					{
						type: NodesActionType.ADD_NODE,
						id: `default-${i}`,
						defaultNode: true,
						x: dNode.x || 0,
						y: dNode.y || 0,
						nodeType: dNode.type,
					},
					{ nodeTypes, portTypes, context },
				);
			}
			return nodes;
		}, {}),
	};
};

const getDefaultData = ({
	node,
	nodeType,
	portTypes,
	context,
}: {
	node: FlumeNode;
	nodeType: NodeType;
	portTypes: PortTypeMap;
	context: unknown;
}): InputData => {
	if (!nodeType) {
		return {};
	}

	const inputs = Array.isArray(nodeType.inputs)
		? nodeType.inputs
		: nodeType.inputs(node.inputData, node.connections, context);

	return inputs.reduce<InputData>((obj, input) => {
		const inputType = portTypes[input.type];
		obj[input.name || inputType.name] = (
			input.controls ||
			inputType.controls ||
			[]
		).reduce<InputData>((obj2, control) => {
			obj2[control.name] = control.defaultValue as ControlData;
			return obj2;
		}, {});
		return obj;
	}, {});
};

type ProposedConnection = { nodeId: string; portName: string };

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
			const inputIsNotConnected =
				!nodes[input.nodeId].connections.inputs[input.portName];
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
