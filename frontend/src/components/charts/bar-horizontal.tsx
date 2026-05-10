"use client";

import { VegaEmbed } from "react-vega";

interface BarHorizontalProps {
    data: Array<{ label: string; percentage: number }>;
    colors: string[];
    className?: string;
    height?: number;
}

export function BarHorizontal({
    data,
    colors,
    className,
    height = 220
}: BarHorizontalProps) {
    const spec = {
        $schema: "https://vega.github.io/schema/vega-lite/v6.json",
        autosize: { contains: "padding" as const, type: "fit" as const },
        background: "transparent",
        config: {
            view: { stroke: "transparent" }
        },
        data: {
            values: data.map((item, index) => ({
                category: index,
                color: colors[index % colors.length] ?? "#8884d8",
                label: item.label,
                percentage: item.percentage
            }))
        },
        encoding: {
            color: {
                field: "color",
                legend: null,
                scale: null,
                type: "nominal" as const
            },
            tooltip: [
                { field: "label", title: "Label", type: "nominal" as const },
                {
                    field: "percentage",
                    format: ".1f",
                    title: "Percentage",
                    type: "quantitative" as const
                }
            ],
            x: {
                axis: null,
                field: "percentage",
                scale: { domain: [0, 100] },
                type: "quantitative" as const
            },
            y: {
                axis: null,
                field: "category",
                type: "ordinal" as const
            }
        },
        mark: "bar" as const,
        width: "container" as const
    };

    return (
        <div
            className={className}
            style={{ height: `${height}px`, width: "100%" }}
        >
            <VegaEmbed spec={spec} options={{ actions: false }} />
        </div>
    );
}
