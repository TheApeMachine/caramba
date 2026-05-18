import * as THREE from "three";
import type { Edge, Node } from "./vector";
import { bodyWindowWorld, innerNodesLayoutBounds } from "./vector";

export type FramedParent = {
	id: number;
	nodes: Node[];
	edges: Edge[];
};

/**
 * PreviewPass — owns the off-screen render target the framed node's body
 * samples. The orthographic aperture matches {@link bodyWindowWorld} aspect
 * and minimally contains the nested layout bounds, aligned with portal
 * compact framing (see `portalCompactScale` / `computePortalFrame` in scene).
 */
export class PreviewPass {
	readonly target: THREE.WebGLRenderTarget;
	readonly camera = new THREE.OrthographicCamera();

	constructor(baseLongEdgePx = 640) {
		const body = bodyWindowWorld();
		const aspect = Math.max(body.width / body.height, 1 / 4096);

		let widthPx = Math.max(64, Math.round(baseLongEdgePx));
		let heightPx = Math.max(64, Math.round(widthPx / aspect));

		if (!Number.isFinite(widthPx) || !Number.isFinite(heightPx)) {
			widthPx = 512;
			heightPx = Math.round(widthPx / aspect);
		}

		this.target = new THREE.WebGLRenderTarget(widthPx, heightPx, {
			minFilter: THREE.LinearFilter,
			magFilter: THREE.LinearFilter,
			format: THREE.RGBAFormat,
			type: THREE.UnsignedByteType,
		});

		this.target.texture.colorSpace = THREE.SRGBColorSpace;
		this.camera.position.z = 10;
	}

	fitToNodes(nodes: { x: number; y: number }[]): void {
		const layout =
			nodes.length === 0
				? {
						minX: 0,
						maxX: 1,
						minY: 0,
						maxY: 1,
						centreX: 0,
						centreY: 0,
						width: 1,
						height: 1,
					}
				: innerNodesLayoutBounds(nodes);

		const body = bodyWindowWorld();

		const bodyAspect = body.width / body.height;

		const halfBBoxW = layout.width * 0.5;

		const halfBBoxH = layout.height * 0.5;

		let paddedHalfW: number;
		let paddedHalfH: number;

		const bboxAspect = halfBBoxW / Math.max(halfBBoxH, 1 / 8192);

		if (bboxAspect > bodyAspect) {
			paddedHalfW = halfBBoxW;

			paddedHalfH = halfBBoxW / bodyAspect;
		} else {
			paddedHalfH = halfBBoxH;

			paddedHalfW = halfBBoxH * bodyAspect;
		}

		const cx = layout.centreX;

		const cy = layout.centreY;

		this.camera.left = cx - paddedHalfW;

		this.camera.right = cx + paddedHalfW;

		this.camera.top = cy + paddedHalfH;

		this.camera.bottom = cy - paddedHalfH;

		this.camera.updateProjectionMatrix();
	}

	dispose(): void {
		this.target.dispose();
	}
}
