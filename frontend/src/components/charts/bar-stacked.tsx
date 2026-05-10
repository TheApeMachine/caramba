"use client";

import { VegaEmbed } from "react-vega";
import Flex from "../ui/flex";

interface BarStackedProps {
    data: Array<Record<string, unknown> & { name: string }>;
    seriesKeys: string[];
    colors: string[];
    className?: string;
    height?: number;
    showLegend?: boolean;
    categoryLabels?: string[];
}

export function BarStacked({
    data,
    seriesKeys,
    colors,
    className,
    height = 300,
    showLegend = true,
    categoryLabels
}: BarStackedProps) {
    // Transform data for Vega-Lite stacked horizontal bar
    const vegaData = data.flatMap((item, categoryIdx) =>
        seriesKeys.map((key) => ({
            category:
                categoryLabels?.[categoryIdx] ??
                item.name ??
                `Category ${categoryIdx}`,
            categoryIndex: categoryIdx,
            series: key,
            value: typeof item[key] === "number" ? item[key] : 0
        }))
    );

    console.log("vegaData", vegaData, seriesKeys);

    // Ensure we have data before rendering
    if (vegaData.length === 0) {
        return null;
    }

    const isSingleCategory = data.length === 1;

    // Calculate optimal height if not provided or too small
    const itemHeight = 40;
    const calculatedHeight = Math.max(height, seriesKeys.length * itemHeight);

    const spec = {
        $schema: "https://vega.github.io/schema/vega-lite/v6.json",
        autosize: {
            contains: "content" as const,
            resize: true,
            type: "fit" as const
        },
        background: "transparent",
        config: {
            axis: {
                domainColor: "#888",
                gridColor: "#333",
                labelColor: "#fff",
                tickColor: "#888",
                titleColor: "#fff"
            },
            legend: {
                labelColor: "#fff",
                labelFontSize: 11,
                labelLimit: 150,
                orient: "top" as const,
                symbolSize: 80,
                symbolType: "circle" as const,
                titleColor: "#fff"
            },
            text: {
                fill: "white",
                fontSize: 12,
                fontWeight: "bold" as const
            },
            view: { stroke: "transparent" }
        },
        data: { values: vegaData },
        encoding: {
            color: {
                field: "series",
                legend: showLegend
                    ? {
                          direction: "vertical" as const,
                          orient: "top" as const,
                          symbolType: "circle" as const,
                          title: null
                      }
                    : null,
                scale: {
                    domain: seriesKeys,
                    range: colors,
                    type: "ordinal" as const
                },
                type: "nominal" as const
            }
        },
        layer: [
            {
                encoding: {
                    tooltip: [
                        {
                            field: "category",
                            title: "Category",
                            type: "nominal" as const
                        },
                        {
                            field: "series",
                            title: "Option",
                            type: "nominal" as const
                        },
                        {
                            field: "value",
                            format: ".1f",
                            title: "Percentage",
                            type: "quantitative" as const
                        }
                    ],
                    x: {
                        aggregate: "sum" as const,
                        axis: null,
                        field: "value",
                        stack: "normalize" as const,
                        type: "quantitative" as const
                    },
                    y: {
                        axis: isSingleCategory
                            ? null
                            : {
                                  labelAngle: 0,
                                  title: null
                              },
                        field: "category",
                        type: "nominal" as const
                    }
                },
                mark: {
                    opacity: 0.9,
                    type: "bar" as const
                }
            },
            {
                encoding: {
                    text: {
                        condition: {
                            test: "datum.value > 5",
                            value: { expr: "format(datum.value, '.0f') + '%'" }
                        },
                        value: ""
                    },
                    x: {
                        aggregate: "sum" as const,
                        axis: null,
                        band: 0.5,
                        field: "value",
                        stack: "normalize" as const,
                        type: "quantitative" as const
                    },
                    y: {
                        axis: null,
                        field: "category",
                        type: "nominal" as const
                    }
                },
                mark: {
                    align: "center" as const,
                    baseline: "middle" as const,
                    type: "text" as const
                }
            }
        ]
    };

    return (
        <Flex.Column
            fullWidth
            align="center"
            justify="center"
            className="flex-1"
            style={{ height: `${calculatedHeight}px` }}
            id={`vega-chart-${Math.random().toString(36).substr(2, 9)}`}
        >
            <VegaEmbed spec={spec} options={{ actions: false }} />
        </Flex.Column>
    );
}
