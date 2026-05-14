import * as THREE from "three";

/*
World-space orthographic camera with tween/snap/fit helpers. Owns its own
state (camX/Y/Zoom) and the THREE.OrthographicCamera. Knows nothing about
materials or rendering.

Coordinate convention: one world unit ≈ one screen pixel at zoom 1, with
+y pointing up. resize() must be called on every viewport size change
*and* every time the camera state changes so the frustum stays in sync.
*/

export type Tween = {
	fromX: number;
	fromY: number;
	fromZoom: number;
	toX: number;
	toY: number;
	toZoom: number;
	start: number;
	duration: number;
};

export const easeInOutCubic = (t: number) =>
	t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;

export class Camera {
	readonly three = new THREE.OrthographicCamera();
	x = 0;
	y = 0;
	zoom = 1;
	tween: Tween | null = null;

	constructor(private container: HTMLElement) {
		this.three.position.z = 10;
	}

	resize(): void {
		const w = Math.max(1, this.container.clientWidth);
		const h = Math.max(1, this.container.clientHeight);
		const halfW = w / 2 / this.zoom;
		const halfH = h / 2 / this.zoom;
		const cam = this.three;
		cam.left = this.x - halfW;
		cam.right = this.x + halfW;
		cam.top = this.y + halfH;
		cam.bottom = this.y - halfH;
		cam.updateProjectionMatrix();
	}

	viewportSize(): { w: number; h: number } {
		return {
			w: Math.max(1, this.container.clientWidth),
			h: Math.max(1, this.container.clientHeight),
		};
	}

	worldFromScreen(
		sx: number,
		sy: number,
		canvas: HTMLCanvasElement,
	): [number, number] {
		const rect = canvas.getBoundingClientRect();
		const nx = ((sx - rect.left) / rect.width) * 2 - 1;
		const ny = -(((sy - rect.top) / rect.height) * 2 - 1);
		const cam = this.three;
		const wx = this.x + (nx * (cam.right - cam.left)) / 2;
		const wy = this.y + (ny * (cam.top - cam.bottom)) / 2;
		return [wx, wy];
	}

	/* Smoothly frame a rect so it occupies `fillFraction` of the smaller dim. */
	frameRect(
		x: number,
		y: number,
		w: number,
		h: number,
		fillFraction = 0.25,
	): void {
		const { w: cw, h: ch } = this.viewportSize();
		const zoom = Math.min(
			8,
			Math.max(0.1, fillFraction * Math.min(cw / w, ch / h)),
		);
		this.tween = {
			fromX: this.x,
			fromY: this.y,
			fromZoom: this.zoom,
			toX: x,
			toY: y,
			toZoom: zoom,
			start: performance.now(),
			duration: 720,
		};
	}

	/* Comfortable framing of an AABB (~60% of smaller dim). */
	fitTarget(
		minX: number,
		minY: number,
		maxX: number,
		maxY: number,
	): { x: number; y: number; zoom: number } {
		const { w: cw, h: ch } = this.viewportSize();
		const w = Math.max(1, maxX - minX);
		const h = Math.max(1, maxY - minY);
		const zoom = Math.min(8, Math.max(0.1, 0.6 * Math.min(cw / w, ch / h)));
		return {
			x: (minX + maxX) * 0.5,
			y: (minY + maxY) * 0.5,
			zoom,
		};
	}

	/*
	Snap to a state immediately, then tween to a target. Used at level
	transitions so the previous viewport content stays put across the
	level swap before easing out to the new framing.
	*/
	snapAndEaseTo(
		fromX: number,
		fromY: number,
		fromZoom: number,
		toX: number,
		toY: number,
		toZoom: number,
		duration = 600,
	): void {
		this.x = fromX;
		this.y = fromY;
		this.zoom = fromZoom;
		this.resize();
		this.tween = {
			fromX,
			fromY,
			fromZoom,
			toX,
			toY,
			toZoom,
			start: performance.now(),
			duration,
		};
	}

	cancelTween(): void {
		this.tween = null;
	}

	zoomAt(
		sx: number,
		sy: number,
		canvas: HTMLCanvasElement,
		deltaY: number,
	): void {
		this.cancelTween();
		const [wx, wy] = this.worldFromScreen(sx, sy, canvas);
		const factor = Math.exp(-deltaY * 0.0015);
		this.zoom = Math.min(8, Math.max(0.1, this.zoom * factor));
		// keep the cursor anchored to the same world point through the zoom
		this.x = wx - (wx - this.x) / factor;
		this.y = wy - (wy - this.y) / factor;
		this.resize();
	}

	panBy(dxPx: number, dyPx: number): void {
		this.x -= dxPx / this.zoom;
		this.y += dyPx / this.zoom;
		this.resize();
	}

	tickTween(now: number): boolean {
		if (!this.tween) return false;
		const t = Math.min(1, (now - this.tween.start) / this.tween.duration);
		const k = easeInOutCubic(t);
		this.x = this.tween.fromX + (this.tween.toX - this.tween.fromX) * k;
		this.y = this.tween.fromY + (this.tween.toY - this.tween.fromY) * k;
		this.zoom =
			this.tween.fromZoom + (this.tween.toZoom - this.tween.fromZoom) * k;
		this.resize();
		if (t >= 1) this.tween = null;
		return true;
	}
}
