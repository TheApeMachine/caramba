import { createStore } from "@tanstack/store";
import * as THREE from "three";

/*
The render map: two RGBA-float textures the GPU samples each frame.

nodesTexture  — one texel per node, .rg = (x, y) in world units,
                .b  = typeId, .a = flags (1.0 = alive)
edgesTexture  — one texel per edge, .rg = fromNodeIndex (as float),
                .ba = toNodeIndex.  Port indices land in a second
                row when we add ports.

Both buffers are sized as 1D rows for simplicity; we'll lift to 2D when
we cross 4096 nodes. The Float32Arrays `nodeData`/`edgeData` and the
DataTextures share the same memory — writes become visible to the GPU
once we set `needsUpdate = true`.
*/
export const MAX_NODES = 1024;
export const MAX_EDGES = 4096;

export const nodeData = new Float32Array(MAX_NODES * 4);
export const nodesTexture = new THREE.DataTexture(
	nodeData,
	MAX_NODES,
	1,
	THREE.RGBAFormat,
	THREE.FloatType,
);
nodesTexture.minFilter = THREE.NearestFilter;
nodesTexture.magFilter = THREE.NearestFilter;
nodesTexture.needsUpdate = true;

export const edgeData = new Float32Array(MAX_EDGES * 4);
export const edgesTexture = new THREE.DataTexture(
	edgeData,
	MAX_EDGES,
	1,
	THREE.RGBAFormat,
	THREE.FloatType,
);
edgesTexture.minFilter = THREE.NearestFilter;
edgesTexture.magFilter = THREE.NearestFilter;
edgesTexture.needsUpdate = true;

/*
edgePathsTexture stores the four corners of each edge's Manhattan route.
Width = MAX_EDGES * 4 texels; layout per edge i: column 4i+0..4i+3,
each texel .rg = (x, y). p0 = source, p1/p2 = the two bend corners,
p3 = sink. The edge shader expands these into 3 line segments.
*/
export const PATH_TEXELS_PER_EDGE = 8;
// 2D layout: x = corner index (0..PATH_TEXELS_PER_EDGE-1), y = edge index.
// 1D would exceed WebGL2's max texture width once MAX_EDGES * texels > 16384.
export const edgePathData = new Float32Array(
	MAX_EDGES * PATH_TEXELS_PER_EDGE * 4,
);
export const edgePathsTexture = new THREE.DataTexture(
	edgePathData,
	PATH_TEXELS_PER_EDGE,
	MAX_EDGES,
	THREE.RGBAFormat,
	THREE.FloatType,
);
edgePathsTexture.minFilter = THREE.NearestFilter;
edgePathsTexture.magFilter = THREE.NearestFilter;
edgePathsTexture.needsUpdate = true;

export const NODE_W = 240;
export const NODE_H = 140;

/*
Port offsets relative to a node's center. v0 has one input + one output
per node, hard-coded. When real node types land, this becomes a registry
entry per type. The values are also referenced by the routing pass so
edges start/end exactly at port positions.

The port disc's center sits ON the node edge (50% inside the body, 50%
bulging out). Y offsets place the input near the top and the output near
the bottom of the node.
*/
export const PORT_RADIUS = 6;
// Ports sit on the left/right edges of the card body band (between header and
// footer). Body UV ≈ [0.16, 0.78]; its centre maps to ~uv.y 0.47, i.e. a small
// downward offset from the node centre in world units.
const BODY_CENTRE_UV_Y = 0.47;
const PORT_Y = (BODY_CENTRE_UV_Y - 0.5) * NODE_H;
// Half-disc notch: the disc's flat edge sits on the card border. The body
// shader uses halfSize = 0.49 and draws a border band a few texels wide just
// inside that edge — we offset by a small slop so the port covers the band
// rather than letting it show through.
const CARD_HALF_W = NODE_W * 0.495;
const PORT_EDGE_SLOP = 0.5;
export const INPUT_OFFSET_X = -CARD_HALF_W - PORT_EDGE_SLOP;
export const INPUT_OFFSET_Y = PORT_Y;
export const OUTPUT_OFFSET_X = CARD_HALF_W + PORT_EDGE_SLOP;
export const OUTPUT_OFFSET_Y = PORT_Y;

export type Edge = { id: number; from: number; to: number };
export type Node = {
	id: number;
	x: number;
	y: number;
	typeId: number;
	nodes: Node[];
	edges: Edge[];
};

type State = {
	nodes: Node[];
	edges: Edge[];
	path: number[]; // node ids from root to the currently-edited level
	framedNodeId: number | null; // when set, that node's children render as a compact preview inside its body
	nextNodeId: number;
	nextEdgeId: number;
};

const initial: State = {
	nodes: [],
	edges: [],
	path: [],
	framedNodeId: null,
	nextNodeId: 0,
	nextEdgeId: 0,
};

/*
At any level, the current view is the (nodes, edges) pair at the end of
`path`. Path = [] means root. Mutating actions clone the spine of `nodes`
along the path and apply the change to the leaf level.
*/
function mutateLevel(
	state: State,
	mutate: (level: { nodes: Node[]; edges: Edge[] }) => {
		nodes: Node[];
		edges: Edge[];
	},
): State {
	if (state.path.length === 0) {
		const next = mutate({ nodes: state.nodes, edges: state.edges });
		return { ...state, nodes: next.nodes, edges: next.edges };
	}

    const rebuild = (nodes: Node[], depth: number): Node[] => {
		const id = state.path[depth];
		return nodes.map((n) => {
			if (n.id !== id) return n;

            if (depth === state.path.length - 1) {
				const next = mutate({ nodes: n.nodes, edges: n.edges });
				return { ...n, nodes: next.nodes, edges: next.edges };
			}

            return { ...n, nodes: rebuild(n.nodes, depth + 1) };
		});
	};

    return { ...state, nodes: rebuild(state.nodes, 0) };
}

export const vectorStore = createStore(initial, (store) => ({
	addNode: (x: number, y: number, typeId = 0) => {
		store.setState((prev) =>
			mutateLevel(
				{ ...prev, nextNodeId: prev.nextNodeId + 1 },
				(lvl) => ({
					nodes: [
						...lvl.nodes,
						{ id: prev.nextNodeId, x, y, typeId, nodes: [], edges: [] },
					],
					edges: lvl.edges,
				}),
			),
		);
	},
	moveNode: (id: number, x: number, y: number) => {
		store.setState((prev) =>
			mutateLevel(prev, (lvl) => ({
				nodes: lvl.nodes.map((n) => (n.id === id ? { ...n, x, y } : n)),
				edges: lvl.edges,
			})),
		);
	},
	addEdge: (from: number, to: number) => {
		store.setState((prev) =>
			mutateLevel(
				{ ...prev, nextEdgeId: prev.nextEdgeId + 1 },
				(lvl) => ({
					nodes: lvl.nodes,
					edges: [...lvl.edges, { id: prev.nextEdgeId, from, to }],
				}),
			),
		);
	},
	enter: (id: number) => {
		store.setState((prev) => ({
			...prev,
			path: [...prev.path, id],
			framedNodeId: null,
		}));
	},
	up: () => {
		store.setState((prev) => ({
			...prev,
			path: prev.path.slice(0, -1),
			framedNodeId: null,
		}));
	},
	setFramed: (id: number | null) => {
		store.setState((prev) => ({ ...prev, framedNodeId: id }));
	},
}));

/*
Resolves the current level's (nodes, edges) from a state by walking `path`.
Used by the buffer sync and by the renderer for read access. Pure.
*/
export function currentLevel(state: {
	nodes: Node[];
	edges: Edge[];
	path: number[];
}): { nodes: Node[]; edges: Edge[] } {
	let nodes = state.nodes;
	let edges = state.edges;
	for (const id of state.path) {
		const next = nodes.find((n) => n.id === id);
		if (!next) return { nodes: [], edges: [] };
		nodes = next.nodes;
		edges = next.edges;
	}
	return { nodes, edges };
}

/*
syncBuffers writes the JS-side node/edge lists into the GPU-facing
Float32Arrays. Runs on every store change. Node id is its index in the
buffer for the lifetime of this session (we don't compact on remove yet).
*/
/*
Populate the GPU input textures from any (nodes, edges) list. Exposed so
the renderer can swap in the framed node's inner graph for a preview pass
without going through the store.
*/
export type PositionTransform = (n: Node) => { x: number; y: number };

export function fillInputBuffers(
	nodes: Node[],
	edges: Edge[],
	transform?: PositionTransform,
): void {
	nodeData.fill(0);
	for (let i = 0; i < nodes.length && i < MAX_NODES; i++) {
		const n = nodes[i];
		const off = i * 4;
		const pos = transform ? transform(n) : { x: n.x, y: n.y };
		nodeData[off + 0] = pos.x;
		nodeData[off + 1] = pos.y;
		nodeData[off + 2] = n.typeId;
		nodeData[off + 3] = 1.0;
	}
	nodesTexture.needsUpdate = true;

	edgeData.fill(0);
	const indexOf = new Map<number, number>();
	for (let i = 0; i < nodes.length; i++) indexOf.set(nodes[i].id, i);
	const edgeCount = Math.min(edges.length, MAX_EDGES);
	for (let i = 0; i < edgeCount; i++) {
		const e = edges[i];
		const fromIdx = indexOf.get(e.from);
		const toIdx = indexOf.get(e.to);
		if (fromIdx === undefined || toIdx === undefined) continue;
		const off = i * 4;
		edgeData[off + 0] = fromIdx;
		edgeData[off + 1] = toIdx;
		edgeData[off + 2] = 1.0;
	}
	edgesTexture.needsUpdate = true;
}

function syncBuffers(state: State): void {
	const level = currentLevel(state);
	fillInputBuffers(level.nodes, level.edges);
}

/*
Path texels are written by the GPU compute pass (see compute.ts), not from
JS. The store subscriber syncs node/edge data into the input textures
above; the renderer triggers the path compute once per frame whenever the
store revision changes.
*/
let revision = 0;

export function getRevision(): number {
	return revision;
}

vectorStore.subscribe(() => {
	revision++;
	syncBuffers(vectorStore.state);
});
