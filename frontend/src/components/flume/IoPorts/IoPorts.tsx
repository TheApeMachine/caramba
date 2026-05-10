import React from "react";
import { createPortal } from "react-dom";
import Connection from "#/components/flume/Connection/Connection";
import Control from "#/components/flume/Control/Control";
import {
	calculateEdgePath,
	getPortRect,
} from "#/components/flume/connectionCalculator";
import { CONNECTIONS_ID } from "#/components/flume/constants";
import {
	ConnectionRecalculateContext,
	ContextContext,
	EditorIdContext,
	NodeDispatchContext,
	PortTypesContext,
	StageContext,
	useEdgeRouting,
} from "#/components/flume/context";
import { NodesActionType } from "#/components/flume/nodesReducer";
import type {
	Colors,
	Connections,
	ControlData,
	Control as ControlType,
	InputData,
	PortType,
	PortTypeMap,
	TransputBuilder,
	TransputType,
} from "#/components/flume/types";
import { Button } from "#/components/ui/button";
import { Field } from "#/components/ui/field";
import { Fieldset } from "#/components/ui/fieldset";
import { Flex } from "#/components/ui/flex";
import usePrevious from "#/hooks/usePrevious";
import { cn } from "@/lib/utils";
import styles from "./IoPorts.module.css";

const useTransputs = (
	transputsFn: PortType[] | TransputBuilder,
	transputType: TransputType,
	nodeId: string,
	inputData: InputData,
	connections: Connections,
) => {
	const nodesDispatch = React.useContext(NodeDispatchContext);
	const executionContext = React.useContext(ContextContext);

	const transputs = React.useMemo(() => {
		if (Array.isArray(transputsFn)) return transputsFn;
		return transputsFn(inputData, connections, executionContext);
	}, [transputsFn, inputData, connections, executionContext]);

	const prevTransputs = usePrevious<PortType[]>(transputs);

	React.useEffect(() => {
		if (!prevTransputs || Array.isArray(transputsFn)) return;

		for (const transput of prevTransputs) {
			const current = transputs.find(({ name }) => transput.name === name);

			if (!current) {
				nodesDispatch?.({
					type: NodesActionType.DESTROY_TRANSPUT,
					transputType,
					transput: { nodeId, portName: `${transput.name}` },
				});
			}
		}
	}, [
		transputsFn,
		transputs,
		prevTransputs,
		nodesDispatch,
		nodeId,
		transputType,
	]);

	return transputs;
};

interface IoPortsProps {
	nodeId: string;
	inputs: PortType[] | TransputBuilder;
	outputs: PortType[] | TransputBuilder;
	connections: Connections;
	inputData: InputData;
	updateNodeConnections: () => void;
}

const IoPorts = ({
	nodeId,
	inputs = [],
	outputs = [],
	connections,
	inputData,
	updateNodeConnections,
}: IoPortsProps) => {
	const inputTypes = React.useContext(PortTypesContext);
	const triggerRecalculation = React.useContext(ConnectionRecalculateContext);
	const resolvedInputs = useTransputs(
		inputs,
		"input",
		nodeId,
		inputData,
		connections,
	);
	const resolvedOutputs = useTransputs(
		outputs,
		"output",
		nodeId,
		inputData,
		connections,
	);

	if (!triggerRecalculation || !inputTypes) {
		return null;
	}

	return (
		<Flex.Column
			padding={1}
			fullWidth
			className="mt-auto"
			data-flume-component="ports"
		>
			{resolvedInputs.length ? (
				<Flex.Column
					align="stretch"
					data-flume-component="ports-inputs"
					fullWidth
					gap={3}
				>
					{resolvedInputs.map((input) => (
						<Input
							{...input}
							data={inputData[input.name] || {}}
							isConnected={!!connections.inputs[input.name]}
							triggerRecalculation={triggerRecalculation ?? (() => {})}
							updateNodeConnections={updateNodeConnections}
							inputTypes={inputTypes ?? {}}
							nodeId={nodeId}
							inputData={inputData}
							key={input.name}
						/>
					))}
				</Flex.Column>
			) : null}
			{!!resolvedOutputs.length && (
				<Flex.Row
					align="center"
					justify="end"
					data-flume-component="ports-outputs"
					fullWidth
				>
					{resolvedOutputs.map((output) => (
						<Output
							{...output}
							triggerRecalculation={triggerRecalculation}
							inputTypes={inputTypes}
							nodeId={nodeId}
							key={output.name}
						/>
					))}
				</Flex.Row>
			)}
		</Flex.Column>
	);
};

export default IoPorts;

interface InputProps {
	type: string;
	label: string;
	name: string;
	nodeId: string;
	data: ControlData;
	controls: ControlType[];
	inputTypes: PortTypeMap;
	noControls?: boolean;
	triggerRecalculation: () => void;
	updateNodeConnections: () => void;
	isConnected?: boolean;
	inputData: InputData;
	hidePort?: boolean;
}

const Input = ({
	type,
	label,
	name,
	nodeId,
	data,
	controls: localControls,
	inputTypes,
	noControls,
	triggerRecalculation,
	updateNodeConnections,
	isConnected,
	inputData,
	hidePort,
}: InputProps) => {
	const {
		label: defaultLabel,
		color,
		controls: defaultControls = [],
	} = inputTypes[type] || {};
	const prevConnected = usePrevious(isConnected);

	const controls = localControls || defaultControls;

	React.useEffect(() => {
		if (isConnected !== prevConnected) {
			triggerRecalculation();
		}
	}, [isConnected, prevConnected, triggerRecalculation]);

	return (
		<fieldset
			data-flume-component="port-input"
			className={cn(styles.transput, "border-0 p-0")}
			data-controlless={isConnected || noControls || !controls.length}
			onDragStart={(e) => {
				e.preventDefault();
				e.stopPropagation();
			}}
		>
			{!hidePort ? (
				<Port
					type={type}
					color={color}
					name={name}
					nodeId={nodeId}
					isInput
					triggerRecalculation={triggerRecalculation}
				/>
			) : null}
			{(!controls.length || noControls || isConnected) && (
				<span data-flume-component="port-label" className={styles.portLabel}>
					{label || defaultLabel}
				</span>
			)}
			{!noControls && !isConnected ? (
				<div className={styles.controls}>
					{(() => {
						const isMono = controls.length === 1;
						return controls.map((control) => {
							switch (control.type) {
								case "text": {
									const value =
										(data[control.name] as string | undefined) ??
										control.defaultValue;
									return (
										<Control
											{...control}
											nodeId={nodeId}
											portName={name}
											triggerRecalculation={triggerRecalculation}
											updateNodeConnections={updateNodeConnections}
											inputLabel={label}
											data={value}
											allData={data}
											key={control.name}
											inputData={inputData}
											isMonoControl={isMono}
										/>
									);
								}
								case "number": {
									const value =
										(data[control.name] as number | undefined) ??
										control.defaultValue;
									return (
										<Control
											{...control}
											nodeId={nodeId}
											portName={name}
											triggerRecalculation={triggerRecalculation}
											updateNodeConnections={updateNodeConnections}
											inputLabel={label}
											data={value}
											allData={data}
											key={control.name}
											inputData={inputData}
											isMonoControl={isMono}
										/>
									);
								}
								case "checkbox": {
									const value =
										(data[control.name] as boolean | undefined) ??
										control.defaultValue;
									return (
										<Control
											{...control}
											nodeId={nodeId}
											portName={name}
											triggerRecalculation={triggerRecalculation}
											updateNodeConnections={updateNodeConnections}
											inputLabel={label}
											data={value}
											allData={data}
											key={control.name}
											inputData={inputData}
											isMonoControl={isMono}
										/>
									);
								}
								case "select": {
									const value =
										(data[control.name] as string | undefined) ??
										control.defaultValue;
									return (
										<Control
											{...control}
											nodeId={nodeId}
											portName={name}
											triggerRecalculation={triggerRecalculation}
											updateNodeConnections={updateNodeConnections}
											inputLabel={label}
											data={value}
											allData={data}
											key={control.name}
											inputData={inputData}
											isMonoControl={isMono}
										/>
									);
								}
								case "multiselect": {
									const value =
										(data[control.name] as string[] | undefined) ??
										control.defaultValue;
									return (
										<Control
											{...control}
											nodeId={nodeId}
											portName={name}
											triggerRecalculation={triggerRecalculation}
											updateNodeConnections={updateNodeConnections}
											inputLabel={label}
											data={value}
											allData={data}
											key={control.name}
											inputData={inputData}
											isMonoControl={isMono}
										/>
									);
								}
								case "custom": {
									const value = data[control.name];
									return (
										<Control
											{...control}
											nodeId={nodeId}
											portName={name}
											triggerRecalculation={triggerRecalculation}
											updateNodeConnections={updateNodeConnections}
											inputLabel={label}
											data={value}
											allData={data}
											key={control.name}
											inputData={inputData}
											isMonoControl={isMono}
										/>
									);
								}
								default:
									return null;
							}
						});
					})()}
				</div>
			) : null}
		</fieldset>
	);
};

interface OutputProps {
	label: string;
	name: string;
	nodeId: string;
	type: string;
	inputTypes: PortTypeMap;
	triggerRecalculation: () => void;
}

const Output = ({
	label,
	name,
	nodeId,
	type,
	inputTypes,
	triggerRecalculation,
}: OutputProps) => {
	const { label: defaultLabel, color } = inputTypes[type] || {};

	return (
		<Fieldset
			data-flume-component="port-output"
			className="flex align-center"
			data-controlless={true}
			onDragStart={(e) => {
				e.preventDefault();
				e.stopPropagation();
			}}
		>
			<Fieldset.Legend>{label || defaultLabel}</Fieldset.Legend>
			<Field className="flex align-center">
				<Port
					type={type}
					name={name}
					color={color}
					nodeId={nodeId}
					triggerRecalculation={triggerRecalculation}
				/>
			</Field>
		</Fieldset>
	);
};

interface PortProps {
	color: Colors;
	name: string;
	type: string;
	isInput?: boolean;
	nodeId: string;
	triggerRecalculation: () => void;
}

const Port = ({
	color = "grey",
	name = "",
	type,
	isInput,
	nodeId,
	triggerRecalculation,
}: PortProps) => {
	const nodesDispatch = React.useContext(NodeDispatchContext);
	const stageState = React.useContext(StageContext) || {
		scale: 1,
		translate: { x: 0, y: 0 },
	};
	const editorId = React.useContext(EditorIdContext);
	const connectionsDomId = `${CONNECTIONS_ID}${editorId}`;
	const inputTypes = React.useContext(PortTypesContext) ?? {};
	const [isDragging, setIsDragging] = React.useState(false);
	const [dragStartCoordinates, setDragStartCoordinates] = React.useState({
		x: 0,
		y: 0,
	});
	const dragStartCoordinatesCache = React.useRef(dragStartCoordinates);
	const port = React.useRef<HTMLButtonElement>(null);
	const line = React.useRef<SVGPathElement>(null);
	const lineInToPort = React.useRef<HTMLDivElement | null>(null);

	const connectionsPortalHost =
		typeof document !== "undefined"
			? document.getElementById(connectionsDomId)
			: null;

	const edgeRouting = useEdgeRouting();

	const byScale = (value: number) => (1 / (stageState?.scale ?? 1)) * value;

	const handleDrag = (e: MouseEvent) => {
		// Match createConnections(): use the connections container (inside the
		// scaled translate tree), not the outer stage — and do not add translate
		// here because getBoundingClientRect() already reflects pan/zoom.
		const conn = document
			.getElementById(connectionsDomId)
			?.getBoundingClientRect() ?? { x: 0, y: 0, width: 0, height: 0 };
		const { x, y, width, height } = conn;
		const halfW = width / 2;
		const halfH = height / 2;

		if (isInput) {
			const to = {
				x: byScale(e.clientX - x - halfW),
				y: byScale(e.clientY - y - halfH),
			};
			lineInToPort.current?.setAttribute(
				"d",
				calculateEdgePath(
					edgeRouting,
					dragStartCoordinatesCache.current,
					to,
				),
			);
		} else {
			const to = {
				x: byScale(e.clientX - x - halfW),
				y: byScale(e.clientY - y - halfH),
			};
			line.current?.setAttribute(
				"d",
				calculateEdgePath(
					edgeRouting,
					dragStartCoordinatesCache.current,
					to,
				),
			);
		}
	};

	const handleDragEnd = (e: MouseEvent) => {
		const droppedOnPort = !!(e.target as HTMLElement)?.dataset?.portName;

		if (isInput) {
			const {
				inputNodeId = "",
				inputPortName = "",
				outputNodeId = "",
				outputPortName = "",
			} = lineInToPort.current?.dataset ?? {};
			nodesDispatch?.({
				type: NodesActionType.REMOVE_CONNECTION,
				input: { nodeId: inputNodeId, portName: inputPortName },
				output: { nodeId: outputNodeId, portName: outputPortName },
			});
			if (droppedOnPort) {
				const {
					portName: connectToPortName,
					nodeId: connectToNodeId,
					portType: connectToPortType,
					portTransputType: connectToTransputType,
				} = (e.target as HTMLElement).dataset;

				if (
					!connectToPortName ||
					!connectToNodeId ||
					!connectToPortType ||
					!connectToTransputType
				) {
					return;
				}

				const isNotSameNode = outputNodeId !== connectToNodeId;

				if (isNotSameNode && connectToTransputType !== "output") {
					const inputWillAcceptConnection =
						inputTypes[connectToPortType]?.acceptTypes?.includes(type);
					if (inputWillAcceptConnection) {
						nodesDispatch?.({
							type: NodesActionType.ADD_CONNECTION,
							input: {
								nodeId: connectToNodeId,
								portName: connectToPortName,
							},
							output: {
								nodeId: outputNodeId,
								portName: outputPortName,
							},
						});
					}
				}
			}
		} else {
			if (droppedOnPort) {
				const {
					portName: inputPortName,
					nodeId: inputNodeId,
					portType: inputNodeType,
					portTransputType: inputTransputType,
				} = (e.target as HTMLElement).dataset;

				if (
					!inputPortName ||
					!inputNodeId ||
					!inputNodeType ||
					!inputTransputType
				) {
					return;
				}

				const isNotSameNode = inputNodeId !== nodeId;
				if (isNotSameNode && inputTransputType !== "output") {
					const inputWillAcceptConnection =
						inputTypes[inputNodeType]?.acceptTypes?.includes(type);

					if (inputWillAcceptConnection) {
						nodesDispatch?.({
							type: NodesActionType.ADD_CONNECTION,
							output: { nodeId, portName: name },
							input: {
								nodeId: inputNodeId,
								portName: inputPortName,
							},
						});
						triggerRecalculation();
					}
				}
			}
		}
		setIsDragging(false);
		document.removeEventListener("mouseup", handleDragEnd);
		document.removeEventListener("mousemove", handleDrag);
	};

	const beginDragFromPort = () => {
		const {
			x: startPortX = 0,
			y: startPortY = 0,
			width: startPortWidth = 0,
			height: startPortHeight = 0,
		} = port.current?.getBoundingClientRect() || {};

		const conn = document
			.getElementById(connectionsDomId)
			?.getBoundingClientRect() || { x: 0, y: 0, width: 0, height: 0 };
		const stageX = conn.x;
		const stageY = conn.y;
		const stageWidth = conn.width;
		const stageHeight = conn.height;
		const stageHalfWidth = stageWidth / 2;
		const stageHalfHeight = stageHeight / 2;

		if (isInput) {
			lineInToPort.current = document.querySelector(
				`[data-input-node-id="${nodeId}"][data-input-port-name="${name}"]`,
			);
			const portIsConnected = !!lineInToPort.current;
			if (
				portIsConnected &&
				lineInToPort.current &&
				lineInToPort.current.parentElement
			) {
				lineInToPort.current.parentElement.style.zIndex = "9999";
				const outputRect = getPortRect(
					lineInToPort.current.dataset.outputNodeId || "",
					lineInToPort.current.dataset.outputPortName || "",
					"output",
				);
				const {
					x: outputPortX = 0,
					y: outputPortY = 0,
					width: outputPortWidth = 0,
					height: outputPortHeight = 0,
				} = outputRect || {};
				const portHalfX = outputPortWidth / 2;
				const portHalfY = outputPortHeight / 2;

				const coordinates = {
					x: byScale(outputPortX - stageX + portHalfX - stageHalfWidth),
					y: byScale(outputPortY - stageY + portHalfY - stageHalfHeight),
				};
				setDragStartCoordinates(coordinates);
				dragStartCoordinatesCache.current = coordinates;
				setIsDragging(true);
				document.addEventListener("mouseup", handleDragEnd);
				document.addEventListener("mousemove", handleDrag);
			}
		} else {
			const coordinates = {
				x: byScale(startPortX - stageX + startPortWidth / 2 - stageHalfWidth),
				y: byScale(startPortY - stageY + startPortHeight / 2 - stageHalfHeight),
			};
			setDragStartCoordinates(coordinates);
			dragStartCoordinatesCache.current = coordinates;
			setIsDragging(true);
			document.addEventListener("mouseup", handleDragEnd);
			document.addEventListener("mousemove", handleDrag);
		}
	};

	const handleDragStart = (e: React.MouseEvent<HTMLButtonElement>) => {
		e.preventDefault();
		e.stopPropagation();
		beginDragFromPort();
	};

	return (
		<React.Fragment>
			<Button
				ref={port}
				type="button"
				size="sm"
				variant="ghost"
				className={cn(
					"relative z-0 h-3 min-h-3 min-w-3 shrink-0 gap-0 rounded-full border-none p-0 shadow-md ring-offset-background [&]:before:shadow-none!",
					"[&]:hover:bg-transparent!",
					"[&]:data-pressed:bg-transparent!",
					styles.port,
				)}
				onMouseDown={handleDragStart}
				onKeyDown={(e) => {
					if (e.key === "Enter" || e.key === " ") {
						e.preventDefault();
						beginDragFromPort();
					}
				}}
				aria-label={`Connect port ${name}`}
				data-port-color={color}
				data-port-name={name}
				data-port-type={type}
				data-port-transput-type={isInput ? "input" : "output"}
				data-node-id={nodeId}
				data-flume-component="port-handle"
				onDragStart={(e) => {
					e.preventDefault();
					e.stopPropagation();
				}}
			/>
			{isDragging && !isInput && connectionsPortalHost
				? createPortal(
						<Connection
							from={dragStartCoordinates}
							to={dragStartCoordinates}
							lineRef={line}
						/>,
						connectionsPortalHost,
					)
				: null}
		</React.Fragment>
	);
};
