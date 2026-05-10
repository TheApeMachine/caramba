"use client";

import { VegaEmbed } from "react-vega";

export interface BarVerticalDatum {
    label: string;
    value: number;
}

interface BarVerticalProps {
    data: BarVerticalDatum[];
    color?: string;
    className?: string;
    height?: number;
}

// Exported for testing / reuse
export const buildBarVerticalSeries = (data: BarVerticalDatum[]) => {
    const xValues: number[] = [];
    const yValues: number[] = [];

    data.forEach((item, index) => {
        const safeValue =
            typeof item.value === "number" && Number.isFinite(item.value)
                ? item.value
                : 0;
        xValues.push(index);
        yValues.push(safeValue);
    });

    const maxValue = yValues.length > 0 ? Math.max(...yValues) : 0;

    return {
        maxValue,
        xValues,
        yValues
    };
};

export function BarVertical({
    data,
    color,
    className,
    height = 400
}: BarVerticalProps) {
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
            view: { stroke: "transparent" }
        },
        data: {
            values: data.map((item, index) => ({
                category: index,
                label: item.label,
                value:
                    typeof item.value === "number" &&
                    Number.isFinite(item.value)
                        ? item.value
                        : 0
            }))
        },
        encoding: {
            color: {
                value: color ?? "oklch(var(--chart-1))"
            },
            tooltip: [
                { field: "label", title: "Label", type: "nominal" as const },
                {
                    field: "value",
                    title: "Value",
                    type: "quantitative" as const
                }
            ],
            x: {
                axis: null,
                field: "category",
                type: "ordinal" as const
            },
            y: {
                axis: {
                    grid: false
                },
                field: "value",
                scale: {
                    domain: [0, Math.max(...data.map((d) => d.value), 1) * 1.1]
                },
                type: "quantitative" as const
            }
        },
        mark: "bar" as const,
        width: "container" as const
    };

    return (
        <div
            className={className}
            style={{
                height: `${height}px`,
                width: "100%"
            }}
        >
            <VegaEmbed spec={spec} options={{ actions: false }} />
        </div>
    );
}
