import React, { type RefObject } from "react";

import type FlumeCache from "#/components/flume/Cache";
import type { EdgeRoutingMode } from "#/components/flume/connectionCalculator";
import type { NodesAction } from "#/components/flume/nodesReducer";
import type {
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

/*
SubGraphContext is set by a block Node when it renders an inline NodeEditor.
It gives the nested editor a callback to write its NodeMap back into the
parent node's subGraph field, keeping the outer graph in sync.
*/
export const SubGraphContext = React.createContext<
	((subGraph: import("./types").NodeMap) => void) | null
>(null);
