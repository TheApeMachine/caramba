"use client";

import { useMemo } from "react";
import { VegaEmbed } from "react-vega";
import { type ChartConfig, ChartContainer } from "@/components/ui/chart";

export type DistributionBucket = {
    label: string; // e.g., "1-3", "4-7", "8-10"
    value: number; // percentage 0..100
};

interface DistributionGroupedProps {
    data: DistributionBucket[];
    className?: string;
}

/**
 * Horizontal diverging stacked bar to summarize distribution buckets.
 * Example buckets: Unfavorable/Neutral/Favorable.
 */
export function DistributionGrouped({
    data,
    className
}: DistributionGroupedProps) {
    const chartConfig: ChartConfig = {
        favorable: { color: "oklch(var(--chart-1))", label: "Favorable" },
        neutral: { color: "oklch(var(--chart-2))", label: "Neutral" },
        unfavorable: { color: "oklch(var(--chart-3))", label: "Unfavorable" }
    } as const;

    // Map into a single row with three keys for stacking
    const [unfavorable = 0, neutral = 0, favorable = 0] = [
        data[0]?.value ?? 0,
        data[1]?.value ?? 0,
        data[2]?.value ?? 0
    ];

    const widths = useMemo(() => {
        const clamp = (v: number) => Math.max(0, Math.min(100, Math.round(v)));
        return {
            favorable: clamp(favorable),
            neutral: clamp(neutral),
            unfavorable: clamp(unfavorable)
        };
    }, [unfavorable, neutral, favorable]);

    const vegaData = [
        {
            category: "distribution",
            series: "unfavorable",
            value: widths.unfavorable
        },
        { category: "distribution", series: "neutral", value: widths.neutral },
        {
            category: "distribution",
            series: "favorable",
            value: widths.favorable
        }
    ];

    const spec = {
        $schema: "https://vega.github.io/schema/vega-lite/v6.json",
        autosize: { contains: "padding" as const, type: "fit" as const },
        background: "transparent",
        config: {
            view: { stroke: "transparent" }
        },
        data: { values: vegaData },
        encoding: {
            color: {
                field: "series",
                legend: null,
                scale: {
                    domain: ["unfavorable", "neutral", "favorable"],
                    range: [
                        chartConfig.unfavorable.color,
                        chartConfig.neutral.color,
                        chartConfig.favorable.color
                    ]
                },
                type: "nominal" as const
            },
            tooltip: [
                { field: "series", title: "Series", type: "nominal" as const },
                {
                    field: "value",
                    format: ".0f",
                    title: "Value",
                    type: "quantitative" as const
                }
            ],
            x: {
                axis: null,
                field: "value",
                scale: { domain: [0, 100] },
                stack: "zero" as const,
                type: "quantitative" as const
            },
            y: {
                axis: null,
                field: "category",
                type: "nominal" as const
            }
        },
        mark: {
            type: "bar" as const
        },
        width: "container" as const
    };

    return (
        <ChartContainer config={chartConfig} className={className}>
            <VegaEmbed spec={spec} options={{ actions: false }} />
        </ChartContainer>
    );
}
