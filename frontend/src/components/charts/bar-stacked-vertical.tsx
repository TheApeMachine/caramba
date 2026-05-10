"use client";

import { VegaEmbed } from "react-vega";

interface BarStackedVerticalProps {
    data: Array<Record<string, unknown>>;
    seriesKeys: string[];
    colors: string[];
    className?: string;
    height?: number;
    showLegend?: boolean;
}

export function BarStackedVertical({
    data,
    seriesKeys,
    colors,
    className,
    height = 300,
    showLegend = true
}: BarStackedVerticalProps) {
    // Transform data for Vega-Lite
    const vegaData = data.flatMap((item, idx) =>
        seriesKeys.map((key) => ({
            category: idx,
            series: key,
            value: typeof item[key] === "number" ? item[key] : 0
        }))
    );

    const spec = {
        $schema: "https://vega.github.io/schema/vega-lite/v6.json",
        autosize: { contains: "padding" as const, type: "fit" as const },
        background: "transparent",
        config: {
            axis: {
                domainColor: "oklch(var(--muted-foreground))",
                gridColor: "oklch(var(--border))",
                labelColor: "oklch(var(--foreground))",
                tickColor: "oklch(var(--muted-foreground))",
                titleColor: "oklch(var(--foreground))"
            },
            legend: {
                labelColor: "oklch(var(--foreground))",
                titleColor: "oklch(var(--foreground))"
            },
            view: { stroke: "transparent" }
        },
        data: { values: vegaData },
        encoding: {
            color: {
                field: "series",
                legend: showLegend
                    ? {
                          direction: "horizontal" as const,
                          orient: "top" as const
                      }
                    : null,
                scale: {
                    domain: seriesKeys,
                    range: colors
                },
                type: "nominal" as const
            },
            tooltip: [
                {
                    field: "category",
                    title: "Category",
                    type: "ordinal" as const
                },
                { field: "series", title: "Series", type: "nominal" as const },
                {
                    field: "value",
                    format: ".1f",
                    title: "Value",
                    type: "quantitative" as const
                }
            ],
            x: {
                axis: {
                    labelAngle: 0,
                    title: null
                },
                field: "category",
                type: "ordinal" as const
            },
            y: {
                axis: {
                    format: "d",
                    title: null
                },
                field: "value",
                scale: { domain: [0, 100] },
                stack: "zero" as const,
                type: "quantitative" as const
            }
        },
        mark: {
            opacity: 0.8,
            type: "bar" as const
        },
        width: "container" as const
    };

    return (
        <div
            className={className}
            style={{ height: `${height}px`, minHeight: "280px", width: "100%" }}
        >
            <VegaEmbed spec={spec} options={{ actions: false }} />
        </div>
    );
}
