import type { Camera } from "./camera";

/*
Mouse/wheel input controller for the scene. Pan only kicks in once the
cursor has moved past a slop threshold; that lets a click-on-empty be
distinguished from a pan-start and prevents click-on-node from being
hijacked when the user's hand wobbles a pixel.

The component layer owns node hit logic via the `pickNode` callback. This
controller does not know about nodes — it just asks "is there a hit
here?" before deciding to pan.
*/

const PAN_SLOP_PX = 5;

export type InputController = {
	dispose: () => void;
};

export function attachInput(
	canvas: HTMLCanvasElement,
	camera: Camera,
	pickNode: (sx: number, sy: number) => number,
	onChange: () => void,
): InputController {
	const pan = {
		armed: false,
		moving: false,
		downX: 0,
		downY: 0,
		lastX: 0,
		lastY: 0,
	};

	const onWheel = (e: WheelEvent) => {
		e.preventDefault();
		camera.zoomAt(e.clientX, e.clientY, canvas, e.deltaY);
		onChange();
	};

	const onMouseDown = (e: MouseEvent) => {
		if (e.button === 1) {
			pan.armed = true;
		} else if (e.button === 0 && pickNode(e.clientX, e.clientY) < 0) {
			pan.armed = true;
		}
		if (!pan.armed) return;
		pan.moving = false;
		pan.downX = e.clientX;
		pan.downY = e.clientY;
		pan.lastX = e.clientX;
		pan.lastY = e.clientY;
	};

	const onMouseMove = (e: MouseEvent) => {
		if (!pan.armed) return;
		if (!pan.moving) {
			const dx = e.clientX - pan.downX;
			const dy = e.clientY - pan.downY;
			if (dx * dx + dy * dy < PAN_SLOP_PX * PAN_SLOP_PX) return;
			pan.moving = true;
			camera.cancelTween();
		}
		const dx = e.clientX - pan.lastX;
		const dy = e.clientY - pan.lastY;
		pan.lastX = e.clientX;
		pan.lastY = e.clientY;
		camera.panBy(dx, dy);
		onChange();
	};

	const onMouseUp = () => {
		pan.armed = false;
		pan.moving = false;
	};

	canvas.addEventListener("wheel", onWheel, { passive: false });
	canvas.addEventListener("mousedown", onMouseDown);
	window.addEventListener("mousemove", onMouseMove);
	window.addEventListener("mouseup", onMouseUp);

	return {
		dispose: () => {
			canvas.removeEventListener("wheel", onWheel);
			canvas.removeEventListener("mousedown", onMouseDown);
			window.removeEventListener("mousemove", onMouseMove);
			window.removeEventListener("mouseup", onMouseUp);
		},
	};
}
