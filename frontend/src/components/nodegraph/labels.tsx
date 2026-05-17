import { useSelector } from "@tanstack/react-store";
import { useEffect, useRef } from "react";
import type { SceneHandle } from "./scene";
import { getNodeType } from "./types";
import { currentLevel, NODE_H, type Node, vectorStore } from "./vector";

// Header band runs from UV.y 0.75 to 1.0; its centre at UV 0.875 maps to
// (0.875 - 0.5) * NODE_H above the node centre in world units.
const HEADER_CENTRE_OFFSET = 0.375 * NODE_H;

interface NodeLabelsProps {
	sceneRef: React.RefObject<SceneHandle | null>;
}

/*
NodeLabels overlays a small text element above each node's header band.
Positions are recomputed every animation frame from the camera state so
labels stick to their nodes through pans, zooms, and unfold animations.
Label content is the operation's `opId` when present (a backend op) or
the local NodeType's `label` otherwise.
*/
export function NodeLabels({ sceneRef }: NodeLabelsProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const refMap = useRef(new Map<number, HTMLDivElement>());

	const nodes = useSelector(vectorStore, (state) => currentLevel(state).nodes);

	useEffect(() => {
		let raf = 0;

		const tick = () => {
			raf = requestAnimationFrame(tick);
			const scene = sceneRef.current;
			if (!scene) return;
			const zoom = scene.getCameraZoom();

			for (const [id, el] of refMap.current) {
				const node = nodes.find((n) => n.id === id);
				if (!node) {
					el.style.display = "none";
					continue;
				}
				const [sx, sy] = scene.screenFromWorld(
					node.x,
					node.y + HEADER_CENTRE_OFFSET,
				);
				el.style.display = "";
				el.style.transform = `translate(-50%, -50%) translate(${sx}px, ${sy}px)`;
				const scale = Math.min(1.6, Math.max(0.55, zoom));
				el.style.fontSize = `${11 * scale}px`;
			}
		};

		raf = requestAnimationFrame(tick);
		return () => cancelAnimationFrame(raf);
	}, [nodes, sceneRef]);

	return (
		<div
			ref={containerRef}
			style={{
				inset: 0,
				pointerEvents: "none",
				position: "absolute",
			}}
		>
			{nodes.map((node) => (
				<div
					key={node.id}
					ref={(el) => {
						if (el) refMap.current.set(node.id, el);
						else refMap.current.delete(node.id);
					}}
					style={{
						color: "#e6e9ef",
						fontFamily: "monospace",
						fontWeight: 600,
						left: 0,
						maxWidth: 220,
						overflow: "hidden",
						position: "absolute",
						textOverflow: "ellipsis",
						textShadow: "0 1px 2px rgba(0,0,0,0.75)",
						top: 0,
						whiteSpace: "nowrap",
					}}
				>
					{labelFor(node)}
				</div>
			))}
		</div>
	);
}

function labelFor(node: Node): string {
	if (node.opId) return node.opId;
	return getNodeType(node.typeId).label;
}
