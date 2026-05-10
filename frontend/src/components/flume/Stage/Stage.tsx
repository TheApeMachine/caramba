import type { MouseEventHandler, RefObject } from "react";
import React from "react";
import { createPortal } from "react-dom";
import type { CommentAction } from "#/components/flume/commentsReducer";
import { CommentActionTypes } from "#/components/flume/commentsReducer";
import { STAGE_ID } from "#/components/flume/constants";
import {
	NodeDispatchContext,
	NodeTypesContext,
} from "#/components/flume/context";
import { NodesActionType } from "#/components/flume/nodesReducer";
import type { StageActionSetter } from "#/components/flume/stageReducer";
import { StageActionType } from "#/components/flume/stageReducer";
import type {
	Coordinate,
	SelectOption,
	StageState,
	StageTranslate,
} from "#/components/flume/types";
import ContextMenu from "../ContextMenu/ContextMenu";
import Draggable from "../Draggable/Draggable";
import styles from "./Stage.module.css";

interface StageProps {
	scale: number;
	translate: StageTranslate;
	editorId: string;
	dispatchStageState: React.Dispatch<StageActionSetter>;
	children: React.ReactNode;
	outerStageChildren: React.ReactNode;
	numNodes: number;
	stageRef: RefObject<DOMRect | undefined>;
	spaceToPan: boolean;
	dispatchComments: React.Dispatch<CommentAction>;
	disableComments: boolean;
	disablePan: boolean;
	disableZoom: boolean;
	disableFocusCapture: boolean;
}

const Stage = ({
	scale,
	translate,
	editorId,
	dispatchStageState,
	children,
	outerStageChildren,
	numNodes,
	stageRef,
	spaceToPan,
	dispatchComments,
	disableComments,
	disablePan,
	disableZoom,
	disableFocusCapture,
}: StageProps) => {
	const nodeTypes = React.useContext(NodeTypesContext);
	const dispatchNodes = React.useContext(NodeDispatchContext);
	const wrapper = React.useRef<HTMLDivElement>(null);
	const translateWrapper = React.useRef<HTMLDivElement>(null);
	const [menuOpen, setMenuOpen] = React.useState(false);
	const [menuCoordinates, setMenuCoordinates] = React.useState({
		x: 0,
		y: 0,
	});
	const dragData = React.useRef({ x: 0, y: 0 });
	const [spaceIsPressed, setSpaceIsPressed] = React.useState(false);

	const setStageRect = React.useCallback(() => {
		if (wrapper.current) {
			stageRef.current = wrapper.current.getBoundingClientRect();
		}
	}, [stageRef]);

	React.useEffect(() => {
		if (wrapper.current) {
			stageRef.current = wrapper.current.getBoundingClientRect();
		}

		window.addEventListener("resize", setStageRect);
		return () => {
			window.removeEventListener("resize", setStageRect);
		};
	}, [stageRef, setStageRect]);

	const handleWheel = React.useCallback(
		(e: WheelEvent) => {
			const wheelTarget = e.target as HTMLElement;
			if (wheelTarget.nodeName === "TEXTAREA" || wheelTarget.dataset.comment) {
				if (wheelTarget.clientHeight < wheelTarget.scrollHeight) return;
			}
			e.preventDefault();
			e.stopPropagation();
			if (numNodes === 0) return;

			const wrapperRect = wrapper.current?.getBoundingClientRect();

			if (wrapperRect) {
				dispatchStageState((stageState: StageState) => {
					const { scale: currentScale, translate: currentTranslate } =
						stageState;
					const delta = e.deltaY;
					const clampedDelta = Math.min(10, Math.max(-10, delta));
					const newScale: number = Math.min(
						7,
						Math.max(0.1, currentScale - clampedDelta * 0.005),
					);

					const byOldScale = (no: number) => no * (1 / currentScale);
					const byNewScale = (no: number) => no * (1 / newScale);

					const xOld = byOldScale(
						e.clientX -
							wrapperRect.x -
							wrapperRect.width / 2 +
							currentTranslate.x,
					);
					const yOld = byOldScale(
						e.clientY -
							wrapperRect.y -
							wrapperRect.height / 2 +
							currentTranslate.y,
					);

					const xNew = byNewScale(
						e.clientX -
							wrapperRect.x -
							wrapperRect.width / 2 +
							currentTranslate.x,
					);
					const yNew = byNewScale(
						e.clientY -
							wrapperRect.y -
							wrapperRect.height / 2 +
							currentTranslate.y,
					);

					const xDistance = xOld - xNew;
					const yDistance = yOld - yNew;

					return {
						type: StageActionType.SET_TRANSLATE_SCALE,
						scale: newScale,
						translate: {
							x: currentTranslate.x + xDistance * newScale,
							y: currentTranslate.y + yDistance * newScale,
						},
					};
				});
			}
		},
		[dispatchStageState, numNodes],
	);

	const handleDragDelayStart = () => {
		wrapper.current?.focus();
	};

	const handleDragStart = (event: MouseEvent | TouchEvent) => {
		const e = event as MouseEvent;
		e.preventDefault();
		dragData.current = {
			x: e.clientX,
			y: e.clientY,
		};
	};

	const handleMouseDrag = (_coords: Coordinate, e: MouseEvent) => {
		const xDistance = dragData.current.x - e.clientX;
		const yDistance = dragData.current.y - e.clientY;
		const xDelta = translate.x + xDistance;
		const yDelta = translate.y + yDistance;
		if (wrapper.current) {
			wrapper.current.style.backgroundPosition = `${-xDelta}px ${-yDelta}px`;
		}
		if (translateWrapper.current) {
			translateWrapper.current.style.transform = `translate(${-(translate.x + xDistance)}px, ${-(translate.y + yDistance)}px)`;
		}
	};

	const handleDragEnd = (e: MouseEvent) => {
		const xDistance = dragData.current.x - e.clientX;
		const yDistance = dragData.current.y - e.clientY;
		dragData.current.x = e.clientX;
		dragData.current.y = e.clientY;
		dispatchStageState(({ translate: tran }) => ({
			type: StageActionType.SET_TRANSLATE,
			translate: {
				x: tran.x + xDistance,
				y: tran.y + yDistance,
			},
		}));
	};

	const handleContextMenu: MouseEventHandler = (e) => {
		e.preventDefault();
		e.stopPropagation();
		setMenuCoordinates({ x: e.clientX, y: e.clientY });
		setMenuOpen(true);
		return false;
	};

	const closeContextMenu = () => {
		setMenuOpen(false);
	};

	const byScale = (value: number) => (1 / scale) * value;

	const addNode = ({ node, internalType }: SelectOption) => {
		const wrapperRect = wrapper.current?.getBoundingClientRect();

		if (wrapperRect) {
			const x =
				byScale(menuCoordinates.x - wrapperRect.x - wrapperRect.width / 2) +
				byScale(translate.x);
			const y =
				byScale(menuCoordinates.y - wrapperRect.y - wrapperRect.height / 2) +
				byScale(translate.y);
			if (internalType === "comment") {
				dispatchComments({
					type: CommentActionTypes.ADD_COMMENT,
					x,
					y,
				});
			} else {
				dispatchNodes?.({
					type: NodesActionType.ADD_NODE,
					x,
					y,
					nodeType: node?.type || "",
				});
			}
		}
	};

	const handleDocumentKeyUp = (e: KeyboardEvent) => {
		if (e.which === 32) {
			setSpaceIsPressed(false);
			document.removeEventListener("keyup", handleDocumentKeyUp);
		}
	};

	const handleKeyDown = (e: React.KeyboardEvent) => {
		if (e.which === 32 && document.activeElement === wrapper.current) {
			e.preventDefault();
			e.stopPropagation();
			setSpaceIsPressed(true);
			document.addEventListener("keyup", handleDocumentKeyUp);
		}
	};

	const handleMouseEnter = () => {
		if (
			!disableFocusCapture &&
			!wrapper.current?.contains(document.activeElement)
		) {
			wrapper.current?.focus({ preventScroll: true });
		}
	};

	React.useEffect(() => {
		if (!disableZoom) {
			const stageWrapper = wrapper.current;
			stageWrapper?.addEventListener("wheel", handleWheel);
			return () => {
				stageWrapper?.removeEventListener("wheel", handleWheel);
			};
		}
	}, [handleWheel, disableZoom]);

	/** Avoid preventDefault/arming stage pan when the gesture starts on subgraph UI or toasts/debug controls. */
	const suppressStageCanvasDragPrep = React.useCallback(
		(e: React.MouseEvent<HTMLDivElement>) =>
			e.target instanceof Element &&
			e.target.closest(
				[
					'[data-flume-component="node"]',
					'[data-flume-component="comment"]',
					"[data-comment]",
					"button",
					"input",
					"textarea",
					"select",
				].join(","),
			) != null,
		[],
	);

	const menuOptions = React.useMemo(() => {
		const options: SelectOption[] = [...Object.values(nodeTypes || {})]
			.filter((node) => node.addable !== false)
			.map((node) => ({
				value: node.type,
				label: node.label,
				description: node.description,
				sortIndex: node.sortIndex,
				category: node.category,
				node,
			}))
			.sort((a, b) => {
				const aIndex = a.sortIndex ?? 0;
				const bIndex = b.sortIndex ?? 0;
				if (aIndex !== bIndex) return aIndex - bIndex;
				return a.label.localeCompare(b.label);
			});
		if (!disableComments) {
			options.push({
				value: "comment",
				label: "Comment",
				description: "A comment for documenting nodes",
				internalType: "comment",
				category: "Canvas",
			});
		}
		return options;
	}, [nodeTypes, disableComments]);

	const portalContainer =
		typeof document !== "undefined" ? document.body : null;

	return (
		<Draggable
			data-flume-component="stage"
			id={`${STAGE_ID}${editorId}`}
			className={styles.wrapper}
			innerRef={wrapper}
			onContextMenu={handleContextMenu}
			onMouseEnter={handleMouseEnter}
			onDragDelayStart={handleDragDelayStart}
			onDragStart={handleDragStart}
			onDrag={handleMouseDrag}
			onDragEnd={handleDragEnd}
			onKeyDown={handleKeyDown}
			tabIndex={-1}
			stageState={{ scale, translate }}
			style={{ cursor: spaceIsPressed && spaceToPan ? "grab" : "" }}
			disabled={disablePan || (spaceToPan && !spaceIsPressed)}
			data-flume-stage={true}
			suppressDragPrep={suppressStageCanvasDragPrep}
		>
			{portalContainer && menuOpen
				? createPortal(
						<ContextMenu
							x={menuCoordinates.x}
							y={menuCoordinates.y}
							options={menuOptions}
							onRequestClose={closeContextMenu}
							onOptionSelected={addNode}
							label="Add Node"
						/>,
						portalContainer,
					)
				: null}
			<div
				ref={translateWrapper}
				className={styles.transformWrapper}
				style={{
					transform: `translate(${-translate.x}px, ${-translate.y}px)`,
				}}
			>
				<div
					className={styles.scaleWrapper}
					style={{ transform: `scale(${scale})` }}
				>
					{children}
				</div>
			</div>
			{outerStageChildren}
		</Draggable>
	);
};
export default Stage;
