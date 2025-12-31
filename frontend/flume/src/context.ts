import type { NodeTypeMap, StageState, PortTypeMap } from "./types";
import type { NodesAction } from "./nodesReducer";
import React, { type RefObject } from "react";
import type FlumeCache from "./Cache";

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
    null
);
export const RecalculateStageRectContext = React.createContext<
    null | (() => void)
>(null);
export const EditorIdContext = React.createContext<string>("");
