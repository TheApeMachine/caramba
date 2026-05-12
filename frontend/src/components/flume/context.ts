import React, { type RefObject } from "react";

import type FlumeCache from "#/components/flume/Cache";
import type { EdgeRoutingMode } from "#/components/flume/connectionCalculator";
import type { NodesAction } from "#/components/flume/nodesReducer";
import type {
	FlumeNode,
	NodeMap,
	NodeTypeMap,
	PortTypeMap,
	StageState,
} from "#/components/flume/types";

/** Current edge path style shared by Connections, temporary drag previews, and {@link IoPorts}. */
export const EdgeRoutingContext = React.createContext<EdgeRoutingMode>("smooth");

export function useEdgeRouting(): EdgeRoutingMode {
	return React.useContext(EdgeRoutingContext) ?? "smooth";
}

export const NodeTypesContext = React.createContext<NodeTypeMap | null>(null);
export const PortTypesContext = React.createContext<PortTypeMap | null>(null);
export const NodeDispatchContext =
	React.createContext<React.Dispatch<NodesAction> | null>(null);
export const ConnectionRecalculateContext = React.createContext<
	(() => void) | null
>(null);
export const ContextContext = React.createContext<unknown>(null);
export const StageContext = React.createContext<StageState | null>(null);
export const CacheContext = React.createContext<RefObject<FlumeCache> | null>(
	null,
);
export const RecalculateStageRectContext = React.createContext<
	null | (() => void)
>(null);
export const EditorIdContext = React.createContext<string>("");

/** Maps node IDs to their full node data for consumers that need the full graph. */
export const NodeMapContext = React.createContext<NodeMap>({});

/*
RecalculateConnectionsWorkerContext is a callback that dispatches a full
connection-path recalculation to the off-thread Web Worker. Nodes call this
after each drag move so path math never blocks the main thread.
*/
export const RecalculateConnectionsWorkerContext = React.createContext<
	((nodes: NodeMap) => void) | null
>(null);

/*
SubGraphContext is set by a block Node when it renders an inline NodeEditor.
It gives the nested editor a callback to write its NodeMap back into the
parent node's subGraph field, keeping the outer graph in sync.
*/
export const SubGraphContext = React.createContext<
	((subGraph: NodeMap) => void) | null
>(null);

// Re-export FlumeNode so callers that import from context don't need a second import.
export type { FlumeNode, NodeMap };
