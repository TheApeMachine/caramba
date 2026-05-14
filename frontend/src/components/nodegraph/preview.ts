import * as THREE from "three";
import { type Edge, NODE_H, NODE_W, type Node } from "./vector";

/**
 * PreviewPass — owns the off-screen render target the framed node's body
 * samples. The renderer chooses a square fit of the inner graph's AABB and
 * the main scene is rendered into it with the framed-node uniform pinned
 * off so we don't sample our own output.
 */
export class PreviewPass {
	readonly target: THREE.WebGLRenderTarget;
	readonly camera = new THREE.OrthographicCamera();

	constructor(size = 512) {
		this.target = new THREE.WebGLRenderTarget(size, size, {
			minFilter: THREE.LinearFilter,
			magFilter: THREE.LinearFilter,
			format: THREE.RGBAFormat,
			type: THREE.UnsignedByteType,
		});

		this.target.texture.colorSpace = THREE.SRGBColorSpace;
		this.camera.position.z = 10;
	}

	fitToNodes(nodes: { x: number; y: number }[]): void {
		let minX = Infinity,
			maxX = -Infinity;

		let minY = Infinity,
			maxY = -Infinity;

		const pad = Math.max(NODE_W, NODE_H);

		for (const n of nodes) {
			if (n.x - NODE_W / 2 < minX) minX = n.x - NODE_W / 2;
			if (n.x + NODE_W / 2 > maxX) maxX = n.x + NODE_W / 2;
			if (n.y - NODE_H / 2 < minY) minY = n.y - NODE_H / 2;
			if (n.y + NODE_H / 2 > maxY) maxY = n.y + NODE_H / 2;
		}

		minX -= pad;
		maxX += pad;
		minY -= pad;
		maxY += pad;

		const cx = (minX + maxX) * 0.5;
		const cy = (minY + maxY) * 0.5;
		const side = Math.max(maxX - minX, maxY - minY, 1);

		this.camera.left = cx - side / 2;
		this.camera.right = cx + side / 2;
		this.camera.top = cy + side / 2;
		this.camera.bottom = cy - side / 2;

		this.camera.updateProjectionMatrix();
	}

	dispose(): void {
		this.target.dispose();
	}
}

export type FramedParent = {
	id: number;
	nodes: Node[];
	edges: Edge[];
};
