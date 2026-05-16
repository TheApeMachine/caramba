import * as THREE from "three";
import { Camera, easeInOutCubic } from "./camera";
import { createPathCompute } from "./compute";
import { attachInput } from "./input";
import { BackgroundLayer } from "./materials/background";
import { EdgeLayer } from "./materials/edge";
import { NodeLayer } from "./materials/node";
import { PickLayer } from "./materials/pick";
import { PortLayer } from "./materials/port";
import { RubberEdgeLayer } from "./materials/rubber";
import { type FramedParent, PreviewPass } from "./preview";
import { portsCompatible } from "./types";
import {
	currentLevel,
	fillInputBuffers,
	getRevision,
	NODE_H,
	NODE_W,
	type Node,
	PORT_RADIUS,
	portDefs,
	type PortKind,
	type PositionTransform,
	portWorld,
	vectorStore,
} from "./vector";

const CLEAR_COLOR = 0x0b0d10;
const PREVIEW_BLEND_RATE = 4.0;
const UNFOLD_DURATION_MS = 700;

/*
Body band inside the card shader: header occupies y >= 0.78, footer y < 0.14.
Body centre maps to NODE_H * (0.46 - 0.5) = -0.04 * NODE_H in world units.
The body window is the rectangle the compact sub-graph layout has to fit
into, expressed in world coordinates relative to the card's centre.
*/
const BODY_WINDOW_W = NODE_W * (1 - 2 * 0.10);
const BODY_WINDOW_H = NODE_H * (0.78 - 0.14) * 0.85;
const BODY_WINDOW_CY = (0.46 - 0.5) * NODE_H;

/*
The portal transition lives in a single shared coordinate frame for both
levels: when descending into parent P, inner-world is treated as parent-
world translated by (P.x, P.y). Inner nodes are rendered at
  compactPos = (P.x + innerNode.x * s, P.y + innerNode.y * s + bodyOffsetY)
during fold; their unfolded positions are
  unfoldedPos = (P.x + innerNode.x, P.y + innerNode.y).
We lerp by unfoldT and the camera tweens in the same frame, so no snap.
At t=1 we shift everything by (-P.x, -P.y) — visually a no-op — and we're
in inner-world.
*/
type PortalFrame = {
	parentX: number;
	parentY: number;
	innerCentreX: number;
	innerCentreY: number;
	scale: number;
};

function computePortalFrame(
	parent: { x: number; y: number; nodes: Node[] },
): PortalFrame {
	if (parent.nodes.length === 0) {
		return {
			parentX: parent.x, parentY: parent.y,
			innerCentreX: 0, innerCentreY: 0, scale: 1,
		};
	}
	let minX = Infinity, maxX = -Infinity;
	let minY = Infinity, maxY = -Infinity;
	for (const n of parent.nodes) {
		if (n.x - NODE_W / 2 < minX) minX = n.x - NODE_W / 2;
		if (n.x + NODE_W / 2 > maxX) maxX = n.x + NODE_W / 2;
		if (n.y - NODE_H / 2 < minY) minY = n.y - NODE_H / 2;
		if (n.y + NODE_H / 2 > maxY) maxY = n.y + NODE_H / 2;
	}
	const aabbW = Math.max(maxX - minX, 1);
	const aabbH = Math.max(maxY - minY, 1);
	const scale = Math.min(BODY_WINDOW_W / aabbW, BODY_WINDOW_H / aabbH);
	return {
		parentX: parent.x,
		parentY: parent.y,
		innerCentreX: (minX + maxX) * 0.5,
		innerCentreY: (minY + maxY) * 0.5,
		scale,
	};
}

/*
Transform that maps an inner node's stored position into the shared
parent-frame world for the given unfoldT.
  k = 1 - unfoldT (1 = fully compact, 0 = fully unfolded).
*/
function transformFor(frame: PortalFrame, unfoldT: number): PositionTransform {
	const k = 1 - unfoldT;
	return (n) => {
		const compactX = frame.parentX + (n.x - frame.innerCentreX) * frame.scale;
		const compactY = frame.parentY + (n.y - frame.innerCentreY) * frame.scale
			+ BODY_WINDOW_CY;
		const unfoldedX = frame.parentX + n.x;
		const unfoldedY = frame.parentY + n.y;
		return {
			x: compactX * k + unfoldedX * (1 - k),
			y: compactY * k + unfoldedY * (1 - k),
		};
	};
}

export type PortRef = { nodeId: number; kind: PortKind; portIdx: number };

export type SceneHandle = {
	renderer: THREE.WebGLRenderer;
	canvas: HTMLCanvasElement;
	worldFromScreen: (sx: number, sy: number) => [number, number];
	pickNode: (sx: number, sy: number) => number;
	pickPort: (sx: number, sy: number) => PortRef | null;
	setRubber: (
		from: PortRef | null,
		cursorWorldX: number,
		cursorWorldY: number,
	) => PortRef | null;
	frameRect: (x: number, y: number, w: number, h: number) => void;
	getZoom: () => number;
	beginEnter: (parentId: number) => boolean;
	beginExit: () => boolean;
	isTransitioning: () => boolean;
	dispose: () => void;
};

export function setupScene(container: HTMLElement): SceneHandle {
	const renderer = new THREE.WebGLRenderer({ antialias: true });
	renderer.debug.checkShaderErrors = true;
	renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
	renderer.setClearColor(CLEAR_COLOR, 1);
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	container.appendChild(renderer.domElement);

	const camera = new Camera(container);
	camera.resize();

	const scene = new THREE.Scene();
	const pathCompute = createPathCompute(renderer);
	const preview = new PreviewPass();
	const bg = new BackgroundLayer();
	// 1x1 placeholder bound to uPreview during the preview pass so WebGL
	// never sees the preview target as both a render destination and a
	// sampled texture in the same draw.
	const placeholderTex = new THREE.DataTexture(
		new Uint8Array([0, 0, 0, 255]),
		1, 1, THREE.RGBAFormat, THREE.UnsignedByteType,
	);
	placeholderTex.needsUpdate = true;
	const nodeLayer = new NodeLayer(preview.target.texture);
	const portLayer = new PortLayer();
	const edgeLayer = new EdgeLayer(pathCompute.texture);
	const rubberLayer = new RubberEdgeLayer();
	const pickLayer = new PickLayer(nodeLayer.geom);

	scene.add(bg.mesh);
	scene.add(nodeLayer.shadowMesh);
	scene.add(edgeLayer.mesh);
	scene.add(rubberLayer.mesh);
	scene.add(nodeLayer.bodyMesh);
	scene.add(portLayer.mesh);

	const syncView = () => {
		nodeLayer.setView(camera.three);
		portLayer.setView(camera.three);
		edgeLayer.setView(camera.three);
		rubberLayer.setView(camera.three);
		pickLayer.setView(camera.three);
		const { w, h } = camera.viewportSize();
		bg.update(camera.x, camera.y, camera.zoom, w, h);
	};

	const onResize = () => {
		const w = Math.max(1, container.clientWidth);
		const h = Math.max(1, container.clientHeight);
		// `true` updates the canvas's CSS size to (w, h); the backing buffer
		// is sized to (w·dpr, h·dpr) by the renderer's pixel ratio. Passing
		// `false` leaves CSS untouched, which lets the canvas overflow at
		// dpr=2 and the visible viewport ends up showing the corner of the
		// render rather than its centre.
		renderer.setSize(w, h, true);
		camera.resize();
		syncView();
	};
	const ro = new ResizeObserver(onResize);
	ro.observe(container);
	onResize();

	const pickNode = (sx: number, sy: number) =>
		pickLayer.pick(renderer, camera.three, sx, sy, CLEAR_COLOR);

	/*
	Port hit-test in screen space. Walks every port on every node at the
	current level (cheap: ports per node ≤ ~8, nodes per level ≤ low
	thousands) and picks the closest one within a generous radius. Returns
	the port reference or null.
	*/
	const pickPort = (sx: number, sy: number): PortRef | null => {
		const [wx, wy] = camera.worldFromScreen(sx, sy, renderer.domElement);
		const lvl = currentLevel(vectorStore.state);
		const r = PORT_RADIUS * 1.1;
		const r2 = r * r;
		let best: { ref: PortRef; d2: number } | null = null;
		for (const node of lvl.nodes) {
			for (const kind of ["in", "out"] as const) {
				const defs = portDefs(node, kind);
				for (let idx = 0; idx < defs.length; idx++) {
					const [px, py] = portWorld(node, kind, idx);
					const dx = wx - px;
					const dy = wy - py;
					const d2 = dx * dx + dy * dy;
					if (d2 > r2) continue;
					if (!best || d2 < best.d2) {
						best = { ref: { nodeId: node.id, kind, portIdx: idx }, d2 };
					}
				}
			}
		}
		return best?.ref ?? null;
	};

	/*
	Rubber-band edge: while dragging from a port, the component drives the
	current cursor position. We snap to a compatible port if the cursor is
	near one, otherwise track the cursor freely.
	*/
	const setRubber = (
		from: PortRef | null,
		cursorWorldX: number,
		cursorWorldY: number,
	): PortRef | null => {
		if (!from) {
			rubberLayer.setActive(false);
			return null;
		}
		const lvl = currentLevel(vectorStore.state);
		const fromNode = lvl.nodes.find((n) => n.id === from.nodeId);
		if (!fromNode) {
			rubberLayer.setActive(false);
			return null;
		}
		const [ax, ay] = portWorld(fromNode, from.kind, from.portIdx);
		const fromType = portDefs(fromNode, from.kind)[from.portIdx]?.type;

		// Snap to nearest compatible port within radius. "Compatible" means
		// opposite kind (out↔in), different node, and matching port types
		// (or one side is `any`).
		const wantKind: PortKind = from.kind === "in" ? "out" : "in";
		const snapR = PORT_RADIUS * 1.8;
		const snapR2 = snapR * snapR;
		let snap: { ref: PortRef; x: number; y: number; d2: number } | null = null;
		for (const node of lvl.nodes) {
			if (node.id === from.nodeId) continue;
			const defs = portDefs(node, wantKind);
			for (let idx = 0; idx < defs.length; idx++) {
				if (fromType && !portsCompatible(fromType, defs[idx].type)) continue;
				const [px, py] = portWorld(node, wantKind, idx);
				const dx = cursorWorldX - px;
				const dy = cursorWorldY - py;
				const d2 = dx * dx + dy * dy;
				if (d2 > snapR2) continue;
				if (!snap || d2 < snap.d2) {
					snap = {
						ref: { nodeId: node.id, kind: wantKind, portIdx: idx },
						x: px,
						y: py,
						d2,
					};
				}
			}
		}
		const bx = snap?.x ?? cursorWorldX;
		const by = snap?.y ?? cursorWorldY;
		rubberLayer.setEndpoints(ax, ay, bx, by);
		rubberLayer.setActive(true);
		return snap?.ref ?? null;
	};

	const input = attachInput(renderer.domElement, camera, pickNode, syncView);

	let raf = 0;
	let lastRevision = -1;
	let lastFrameMs = performance.now();
	let previewBlend = 0;

	const computeFramedParent = (): { slot: number; parent: FramedParent | null } => {
		const state = vectorStore.state;
		if (state.framedNodeId === null) return { slot: -1, parent: null };
		const lvl = currentLevel(state);
		const idx = lvl.nodes.findIndex((n) => n.id === state.framedNodeId);
		if (idx < 0) return { slot: -1, parent: null };
		const node = lvl.nodes[idx];
		if (node.nodes.length === 0) return { slot: -1, parent: null };
		return { slot: idx, parent: { id: node.id, nodes: node.nodes, edges: node.edges } };
	};

	const renderPreviewPass = (parent: FramedParent, slot: number) => {
		// Avoid a framebuffer-vs-texture feedback loop: while rendering
		// into preview.target, the node material must not have that same
		// texture bound. Swap to the placeholder for the pass, restore
		// after.
		nodeLayer.setPreview(-1, previewBlend);
		nodeLayer.setPreviewTexture(placeholderTex);
		fillInputBuffers(parent.nodes, parent.edges);
		pathCompute.run();
		preview.fitToNodes(parent.nodes);
		nodeLayer.setView(preview.camera);
		portLayer.setView(preview.camera);
		edgeLayer.setView(preview.camera);
		renderer.setRenderTarget(preview.target);
		renderer.clear();
		renderer.render(scene, preview.camera);
		renderer.setRenderTarget(null);

		// Restore outer-level data + preview sampler + framed slot for the
		// main render. Without this the body shader never knows which
		// instance should composite the preview.
		nodeLayer.setPreviewTexture(preview.target.texture);
		nodeLayer.setPreview(slot, previewBlend);
		const lvl = currentLevel(vectorStore.state);
		fillInputBuffers(lvl.nodes, lvl.edges, activeTransform());
		pathCompute.run();
		syncView();
	};

	/*
	Unfold state. While transitionPhase is non-idle, the inner level's nodes
	are rendered at lerp(compact, full, unfoldT). On "enter" the camera tween
	and unfoldT both run 0 → 1 in parallel. On "exit" we run unfoldT 1 → 0
	*first*, then pop the level (so the camera ease-out from snapAndEaseTo
	already has the now-collapsed sub-graph behind it).
	*/
	type Phase = "idle" | "enter" | "exit";
	let phase: Phase = "idle";
	let phaseStart = 0;
	let unfoldT = 1; // 1 = fully expanded (top-level default)
	// During a transition, both levels share inner-world coordinates: the
	// parent's centre is at the origin (parentX = parentY = 0). At enter
	// start, we shift the camera so visually nothing changes despite the
	// level swap. At exit end, we shift back when popping.
	let portalFrame: PortalFrame = {
		parentX: 0, parentY: 0,
		innerCentreX: 0, innerCentreY: 0, scale: 1,
	};
	let pendingExit = false;
	// Cached parent world position so exit can shift the camera back into
	// parent-world without re-reading the (now-popped) store level.
	let parentWorldX = 0;
	let parentWorldY = 0;

	const activeTransform = (): PositionTransform | undefined => {
		if (phase === "idle" || unfoldT >= 0.999) return undefined;
		return transformFor(portalFrame, unfoldT);
	};

	const beginEnter = (parentId: number): boolean => {
		if (phase !== "idle") return false;
		const outerLvl = currentLevel(vectorStore.state);
		const parent = outerLvl.nodes.find((n) => n.id === parentId);
		if (!parent || parent.nodes.length === 0) {
			vectorStore.actions.enter(parentId);
			return true;
		}

		// Capture the parent's world position before swapping levels.
		parentWorldX = parent.x;
		parentWorldY = parent.y;

		// Inner AABB in inner-world (where the parent's centre is at origin).
		let minX = Infinity, maxX = -Infinity;
		let minY = Infinity, maxY = -Infinity;
		for (const c of parent.nodes) {
			if (c.x - NODE_W / 2 < minX) minX = c.x - NODE_W / 2;
			if (c.x + NODE_W / 2 > maxX) maxX = c.x + NODE_W / 2;
			if (c.y - NODE_H / 2 < minY) minY = c.y - NODE_H / 2;
			if (c.y + NODE_H / 2 > maxY) maxY = c.y + NODE_H / 2;
		}
		const pad = Math.max(NODE_W, NODE_H) * 0.5;
		minX -= pad; maxX += pad; minY -= pad; maxY += pad;

		// Swap level. Inner-world's origin sits at parent's (Px,Py) in
		// parent-world. Translate the camera by -(Px,Py) so the same pixels
		// remain on screen.
		vectorStore.actions.enter(parentId);
		camera.x -= parent.x;
		camera.y -= parent.y;
		camera.resize();

		portalFrame = computePortalFrame({
			x: 0, y: 0, nodes: parent.nodes,
		});

		const fit = camera.fitTarget(minX, minY, maxX, maxY);
		camera.snapAndEaseTo(
			camera.x, camera.y, camera.zoom,
			fit.x, fit.y, fit.zoom,
			UNFOLD_DURATION_MS,
		);
		unfoldT = 0;
		phase = "enter";
		phaseStart = performance.now();
		return true;
	};

	const beginExit = (): boolean => {
		if (phase !== "idle") return false;
		if (vectorStore.state.path.length === 0) return false;
		const lvl = currentLevel(vectorStore.state);
		// We're currently in inner-world. Parent's centre is at origin.
		// Look up the parent in the outer level (one above current path tip)
		// so we can shift the camera back into parent-world at exit end.
		const path = vectorStore.state.path;
		let outerNodes = vectorStore.state.nodes;
		for (let i = 0; i < path.length - 1; i++) {
			const next = outerNodes.find((n) => n.id === path[i]);
			if (!next) break;
			outerNodes = next.nodes;
		}
		const parent = outerNodes.find(
			(n) => n.id === path[path.length - 1],
		);
		parentWorldX = parent?.x ?? 0;
		parentWorldY = parent?.y ?? 0;

		portalFrame = computePortalFrame({
			x: 0, y: 0, nodes: lvl.nodes,
		});

		// Camera target: fit the body window rect around the origin so the
		// fully-folded sub-graph lands inside it.
		const halfW = BODY_WINDOW_W * 0.5;
		const halfH = BODY_WINDOW_H * 0.5;
		const cy = BODY_WINDOW_CY;
		const fit = camera.fitTarget(-halfW, cy - halfH, halfW, cy + halfH);
		camera.snapAndEaseTo(
			camera.x, camera.y, camera.zoom,
			fit.x, fit.y, fit.zoom,
			UNFOLD_DURATION_MS,
		);
		unfoldT = 1;
		phase = "exit";
		phaseStart = performance.now();
		pendingExit = true;
		return true;
	};

	const tickPhase = (now: number) => {
		if (phase === "idle") return false;
		const t = Math.min(1, (now - phaseStart) / UNFOLD_DURATION_MS);
		const k = easeInOutCubic(t);
		unfoldT = phase === "enter" ? k : 1 - k;
		if (t >= 1) {
			if (phase === "exit" && pendingExit) {
				pendingExit = false;
				// Pop level. Inner-world's origin maps back to parent-world's
				// (Px,Py), so shift the camera by +(Px,Py) — same pixels stay
				// on screen.
				vectorStore.actions.up();
				camera.x += parentWorldX;
				camera.y += parentWorldY;
				camera.resize();
			}
			phase = "idle";
			unfoldT = 1;
		}
		return true;
	};

	const animate = () => {
		raf = requestAnimationFrame(animate);
		const now = performance.now();
		const dt = Math.min(0.1, (now - lastFrameMs) / 1000);
		lastFrameMs = now;

		const cameraMoved = camera.tickTween(now);
		const phaseAdvanced = tickPhase(now);
		if (cameraMoved || phaseAdvanced) syncView();

		const { slot, parent } = computeFramedParent();
		const blendTarget = parent ? 1 : 0;
		previewBlend += (blendTarget - previewBlend) * Math.min(1, dt * PREVIEW_BLEND_RATE);
		nodeLayer.setPreview(slot, previewBlend);

		if (parent) {
			renderPreviewPass(parent, slot);
			lastRevision = getRevision();
		} else if (phaseAdvanced) {
			// Unfold animation in progress — refill positions every frame
			// and reroute edges so they follow the moving nodes.
			const lvl = currentLevel(vectorStore.state);
			fillInputBuffers(lvl.nodes, lvl.edges, activeTransform());
			pathCompute.run();
			lastRevision = getRevision();
		} else {
			const rev = getRevision();
			if (rev !== lastRevision) {
				pathCompute.run();
				lastRevision = rev;
			}
		}

		renderer.render(scene, camera.three);
	};
	animate();

	const worldFromScreen = (sx: number, sy: number) =>
		camera.worldFromScreen(sx, sy, renderer.domElement);

	const dispose = () => {
		cancelAnimationFrame(raf);
		ro.disconnect();
		input.dispose();
		pathCompute.dispose();
		preview.dispose();
		placeholderTex.dispose();
		bg.dispose();
		nodeLayer.dispose();
		portLayer.dispose();
		edgeLayer.dispose();
		rubberLayer.dispose();
		pickLayer.dispose();
		renderer.dispose();
		if (renderer.domElement.parentElement === container) {
			container.removeChild(renderer.domElement);
		}
	};

	return {
		renderer,
		canvas: renderer.domElement,
		worldFromScreen,
		pickNode,
		pickPort,
		setRubber,
		frameRect: (x, y, w, h) => camera.frameRect(x, y, w, h),
		getZoom: () => camera.zoom,
		beginEnter,
		beginExit,
		isTransitioning: () => phase !== "idle",
		dispose,
	};
}
