import { useSelector } from "@tanstack/react-store";
import { useEffect, useEffectEvent, useRef } from "react";
import { type PortRef, type SceneHandle, setupScene } from "./scene";
import { currentLevel, NODE_H, NODE_W, vectorStore } from "./vector";

type Mode =
	| { kind: "idle" }
	| {
			kind: "drag-node";
			nodeId: number;
			offsetX: number;
			offsetY: number;
			downX: number;
			downY: number;
	  }
	| { kind: "draw-edge"; from: PortRef };

// Allow this many pixels of mouse jitter before treating it as a drag.
const CLICK_SLOP_PX = 5;

/* When a node is framed and the camera zoom rises past this, descend into
   the framed node. */
const ENTER_ZOOM_THRESHOLD = 3.0;
/* When inside a sub-graph and the camera zoom drops below this, pop back
   to the parent level. */
const EXIT_ZOOM_THRESHOLD = 0.5;
/* After any auto enter/exit, ignore the watcher for this long so the new
   level's zoom doesn't immediately re-trip the same condition. */
const LEVEL_CHANGE_COOLDOWN_MS = 350;

export const NodeGraph = () => {
	const containerRef = useRef<HTMLDivElement>(null);
	const sceneRef = useRef<SceneHandle | null>(null);
	const modeRef = useRef<Mode>({ kind: "idle" });
	const framedNodeRef = useRef<number | null>(null);
	const dragMovedRef = useRef(false);

	const level = useSelector(vectorStore, (s) =>
		currentLevel(s),
	);
	const path = useSelector(vectorStore, (s) => s.path);

	const onMouseDown = useEffectEvent((evt: MouseEvent) => {
		const scene = sceneRef.current;
		if (!scene) return;
		if (evt.target !== scene.canvas) return;
		if (evt.button !== 0) return;

		// Port hit-test wins over node hit-test so the user can drag from a
		// port that sits on the body's edge.
		const port = scene.pickPort(evt.clientX, evt.clientY);
		if (port) {
			modeRef.current = { kind: "draw-edge", from: port };
			const [wx, wy] = scene.worldFromScreen(evt.clientX, evt.clientY);
			scene.setRubber(port, wx, wy);
			return;
		}

		const pickedIdx = scene.pickNode(evt.clientX, evt.clientY);
		if (pickedIdx < 0) return; // empty canvas → scene handles pan

		const [wx, wy] = scene.worldFromScreen(evt.clientX, evt.clientY);
		const lvl = currentLevel(vectorStore.state);
		const node = lvl.nodes[pickedIdx];
		if (!node) return;

		modeRef.current = {
			kind: "drag-node",
			nodeId: node.id,
			offsetX: wx - node.x,
			offsetY: wy - node.y,
			downX: evt.clientX,
			downY: evt.clientY,
		};
		dragMovedRef.current = false;
	});

	const onMouseMove = useEffectEvent((evt: MouseEvent) => {
		const scene = sceneRef.current;
		const mode = modeRef.current;
		if (!scene) return;

		if (mode.kind === "draw-edge") {
			const [wx, wy] = scene.worldFromScreen(evt.clientX, evt.clientY);
			scene.setRubber(mode.from, wx, wy);
			return;
		}

		if (mode.kind !== "drag-node") return;
		// Ignore sub-pixel jitter so a normal click isn't classified as a drag.
		const dx = evt.clientX - mode.downX;
		const dy = evt.clientY - mode.downY;
		if (!dragMovedRef.current && dx * dx + dy * dy < CLICK_SLOP_PX * CLICK_SLOP_PX) {
			return;
		}
		const [wx, wy] = scene.worldFromScreen(evt.clientX, evt.clientY);
		vectorStore.actions.moveNode(
			mode.nodeId,
			wx - mode.offsetX,
			wy - mode.offsetY,
		);
		dragMovedRef.current = true;
	});

	const onMouseUp = useEffectEvent((evt: MouseEvent) => {
		const scene = sceneRef.current;
		const mode = modeRef.current;
		modeRef.current = { kind: "idle" };
		if (!scene) return;

		if (mode.kind === "draw-edge") {
			// Snap target = whatever setRubber would resolve at this cursor pos.
			const [wx, wy] = scene.worldFromScreen(evt.clientX, evt.clientY);
			const snapped = scene.setRubber(mode.from, wx, wy);
			scene.setRubber(null, 0, 0); // clear the rubber-band
			const target = snapped ?? scene.pickPort(evt.clientX, evt.clientY);
			if (!target) return;
			if (target.nodeId === mode.from.nodeId) return;
			if (target.kind === mode.from.kind) return; // strict out↔in
			// Orient so 'from' is always the output port.
			const out = mode.from.kind === "out" ? mode.from : target;
			const inp = mode.from.kind === "in" ? mode.from : target;
			vectorStore.actions.addEdge(out.nodeId, inp.nodeId);
			return;
		}

		if (mode.kind === "drag-node" && !dragMovedRef.current) {
			// Pure click on a node → frame it.
			const lvl = currentLevel(vectorStore.state);
			const node = lvl.nodes.find((n) => n.id === mode.nodeId);
			if (!node) return;
			framedNodeRef.current = node.id;
			vectorStore.actions.setFramed(node.id);
			scene.frameRect(node.x, node.y, NODE_W, NODE_H);
		}
	});

	const onDoubleClick = useEffectEvent((evt: MouseEvent) => {
		const scene = sceneRef.current;
		if (!scene) return;
		const pickedIdx = scene.pickNode(evt.clientX, evt.clientY);
		if (pickedIdx < 0) {
			const [wx, wy] = scene.worldFromScreen(evt.clientX, evt.clientY);
			vectorStore.actions.addNode(wx, wy);
			return;
		}
		const lvl = currentLevel(vectorStore.state);
		const node = lvl.nodes[pickedIdx];
		if (!node) return;
		scene.beginEnter(node.id);
	});

	const onKeyDown = useEffectEvent((evt: KeyboardEvent) => {
		if (evt.key === "Escape") {
			const scene = sceneRef.current;
			framedNodeRef.current = null;
			vectorStore.actions.setFramed(null);
			scene?.beginExit();
		}
	});

	useEffect(() => {
		if (!containerRef.current) return;
		const scene = setupScene(containerRef.current);
		sceneRef.current = scene;

		// One-time seed so a fresh load has something visible at every
		// level we'll demo: two top-level nodes, each with a 3-node
		// sub-graph wired up.
		if (
			vectorStore.state.path.length === 0 &&
			vectorStore.state.nodes.length === 0
		) {
			const { actions } = vectorStore;
			actions.addNode(-200, 0);
			actions.addNode(200, 0);
			actions.addEdge(0, 1);

			for (const parentId of [0, 1]) {
				actions.enter(parentId);
				actions.addNode(-180, -60);
				actions.addNode(0, 60);
				actions.addNode(180, -60);
				const inner = currentLevel(vectorStore.state).nodes;
				actions.addEdge(inner[0].id, inner[1].id);
				actions.addEdge(inner[1].id, inner[2].id);
				actions.up();
			}
		}

		scene.canvas.addEventListener("mousedown", onMouseDown);
		scene.canvas.addEventListener("dblclick", onDoubleClick);
		window.addEventListener("mousemove", onMouseMove);
		window.addEventListener("mouseup", onMouseUp);
		window.addEventListener("keydown", onKeyDown);

		// Zoom watcher: descend on framed + zoom-in, ascend on zoom-out
		// while inside a sub-graph. A cooldown prevents the new level's
		// zoom from immediately re-tripping the opposite threshold.
		let zoomRaf = 0;
		let cooldownUntil = 0;
		const checkZoom = () => {
			zoomRaf = requestAnimationFrame(checkZoom);
			if (performance.now() < cooldownUntil) return;
			if (scene.isTransitioning()) return;
			const zoom = scene.getZoom();
			const id = framedNodeRef.current;
			if (id !== null && zoom >= ENTER_ZOOM_THRESHOLD) {
				framedNodeRef.current = null;
				if (scene.beginEnter(id)) {
					cooldownUntil = performance.now() + LEVEL_CHANGE_COOLDOWN_MS;
				}
				return;
			}
			if (
				vectorStore.state.path.length > 0 &&
				zoom <= EXIT_ZOOM_THRESHOLD
			) {
				if (scene.beginExit()) {
					cooldownUntil = performance.now() + LEVEL_CHANGE_COOLDOWN_MS;
				}
			}
		};
		checkZoom();

		return () => {
			cancelAnimationFrame(zoomRaf);
			scene.canvas.removeEventListener("mousedown", onMouseDown);
			scene.canvas.removeEventListener("dblclick", onDoubleClick);
			window.removeEventListener("mousemove", onMouseMove);
			window.removeEventListener("mouseup", onMouseUp);
			window.removeEventListener("keydown", onKeyDown);
			scene.dispose();
			sceneRef.current = null;
		};
	}, []);

	return (
		<div
			ref={containerRef}
			style={{
				position: "relative",
				width: "100%",
				height: "100%",
				minHeight: "75vh",
				background: "#111",
			}}
		>
			<div
				style={{
					position: "absolute",
					top: 8,
					left: 12,
					color: "#cfd3da",
					fontFamily: "monospace",
					fontSize: 12,
					pointerEvents: "none",
					textShadow: "0 1px 2px rgba(0,0,0,0.6)",
				}}
			>
				path: root{path.map((id) => ` › #${id}`).join("")}
				<br />
				nodes: {level.nodes.length} edges: {level.edges.length}
				<br />
				drag empty → pan · click node → frame · drag node → move ·
				drag from port → edge · double-click node → enter ·
				double-click empty → add node · wheel → zoom · Esc → up
			</div>
		</div>
	);
};
