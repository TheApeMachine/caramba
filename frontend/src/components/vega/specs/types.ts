import type { VisualizationSpec } from "vega-embed";

/*
SeriesPoint is the shared shape for time-series data across spec factories
(one date plus arbitrary numeric series keyed by name).
*/
export type SeriesPoint = { date: number } & Record<string, number>;

export type Spec = VisualizationSpec;
