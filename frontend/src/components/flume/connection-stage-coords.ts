import {
	CANVAS_ID,
	CONNECTIONS_ID,
	STAGE_ID,
} from "#/components/flume/constants";
import type { Coordinate } from "#/components/flume/types";

export const getCanvasRef = (editorId: string) =>
	document.getElementById(`${CANVAS_ID}${editorId}`);

export const getStageRef = (editorId: string) =>
	document.getElementById(
		`${CONNECTIONS_ID}${editorId}`,
	) as HTMLDivElement | null;

/*
getStageBounds returns the visible stage rect used to convert screen
coordinates into the editor's center-origin canvas space.
*/
export const getStageBounds = (editorId: string): DOMRect | null => {
	const stage = document.getElementById(`${STAGE_ID}${editorId}`);
	return stage?.getBoundingClientRect() ?? null;
};

export const screenPointToCanvas = (
	screenX: number,
	screenY: number,
	stageRect: DOMRect,
	scale: number,
): Coordinate => {
	const byScale = (value: number) => (1 / scale) * value;
	const stageHalfWidth = stageRect.width / 2;
	const stageHalfHeight = stageRect.height / 2;

	return {
		x: byScale(screenX - stageRect.x - stageHalfWidth),
		y: byScale(screenY - stageRect.y - stageHalfHeight),
	};
};

export const screenRectToCanvas = (
	rect: DOMRect,
	stageRect: DOMRect,
	scale: number,
): { x: number; y: number; width: number; height: number } => {
	const topLeft = screenPointToCanvas(rect.left, rect.top, stageRect, scale);
	const bottomRight = screenPointToCanvas(
		rect.right,
		rect.bottom,
		stageRect,
		scale,
	);

	return {
		x: topLeft.x,
		y: topLeft.y,
		width: bottomRight.x - topLeft.x,
		height: bottomRight.y - topLeft.y,
	};
};

/*
readLiveStageScale reads the CSS scale from the canvas transform style,
which is more up-to-date than React state during wheel-zoom interactions.
*/
export const readLiveStageScale = (
	editorId: string,
	fallbackScale: number,
): number => {
	const canvas = getCanvasRef(editorId);
	if (!canvas) return fallbackScale;

	const match = canvas.style.transform.match(/scale\(([^)]+)\)/);
	if (!match) return fallbackScale;

	const parsedScale = Number.parseFloat(match[1]);
	if (Number.isNaN(parsedScale) || parsedScale <= 0) return fallbackScale;

	return parsedScale;
};
