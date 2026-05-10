"use client";

import { TrendingUp } from "lucide-react";
import { VegaEmbed } from "react-vega";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { type ChartConfig, ChartContainer } from "@/components/ui/chart";

interface ChartRadarGridFillProps {
    title: string;
    description: string;
}

const chartData = [
    { desktop: 186, month: "Timeline" },
    { desktop: 285, month: "Chats" },
    { desktop: 237, month: "Comments" },
    { desktop: 203, month: "Likes" },
    { desktop: 209, month: "Shares" },
    { desktop: 264, month: "Replies" }
];

const chartConfig = {
    desktop: {
        color: "oklch(var(--chart-1))",
        label: "Desktop"
    }
} satisfies ChartConfig;

export const ChartRadarGridFill = (_: ChartRadarGridFillProps) => {
    const labels = chartData.map((d) => d.month);
    const values = chartData.map((d) => d.desktop);

    // Close the loop for radar chart
    const closedLabels = [...labels, labels[0]];
    const closedValues = [...values, values[0]];

    const spec = {
        $schema: "https://vega.github.io/schema/vega-lite/v6.json",
        autosize: { contains: "padding" as const, type: "fit" as const },
        background: "transparent",
        config: {
            view: { stroke: "transparent" }
        },
        data: {
            values: closedLabels.map((label, idx) => ({
                category: label,
                order: idx,
                value: closedValues[idx]
            }))
        },
        encoding: {
            radius: {
                field: "value",
                scale: {
                    type: "linear" as const,
                    zero: true
                },
                type: "quantitative" as const
            },
            theta: {
                field: "order",
                scale: {
                    domain: [0, labels.length]
                },
                type: "quantitative" as const
            },
            tooltip: [
                {
                    field: "category",
                    title: "Category",
                    type: "nominal" as const
                },
                {
                    field: "value",
                    title: "Value",
                    type: "quantitative" as const
                }
            ]
        },
        mark: {
            color: chartConfig.desktop.color,
            point: true,
            type: "line" as const
        },
        width: "container" as const
    };

    return (
        <Card className="pt-0 gap-0">
            <CardContent className="p-4 py-0">
                <ChartContainer
                    config={chartConfig}
                    className="mx-auto aspect-square h-full"
                >
                    <VegaEmbed spec={spec} options={{ actions: false }} />
                </ChartContainer>
            </CardContent>
            <CardFooter className="flex-col gap-2 text-sm">
                <div className="flex items-center gap-2 leading-none font-medium">
                    Trending up by 5.2% this month{" "}
                    <TrendingUp className="h-4 w-4" />
                </div>
                <div className="text-muted-foreground flex items-center gap-2 leading-none">
                    January - June 2024
                </div>
            </CardFooter>
        </Card>
    );
};

interface RadarComparisonDatum {
    category: string;
    group: number;
    average?: number;
}

interface ChartRadarComparisonProps {
    data: RadarComparisonDatum[];
    groupLabel: string;
    averageLabel?: string;
    className?: string;
    maxChartSize?: number;
}

/**
 * Radar chart comparing a group's engagement metrics vs the average of other groups.
 */
export const ChartRadarComparison = ({
    data,
    groupLabel,
    averageLabel
}: ChartRadarComparisonProps) => {
    const config = {
        average: {
            color: "oklch(var(--chart-2))",
            label: averageLabel ?? "Average"
        },
        group: { color: "oklch(var(--chart-1))", label: groupLabel }
    } satisfies ChartConfig;

    const hasAverage =
        averageLabel && data.some((d) => typeof d.average === "number");

    const labels = data.map((d) => d.category);

    // Transform data for Vega-Lite
    const vegaData = [
        ...data.map((d, idx) => ({
            category: d.category,
            order: idx,
            series: groupLabel,
            value: d.group
        })),
        ...(hasAverage
            ? data.map((d, idx) => ({
                  category: d.category,
                  order: idx,
                  series: averageLabel ?? "Average",
                  value: d.average ?? 0
              }))
            : [])
    ];

    // Close the loop
    const closedData = [...vegaData, ...vegaData.filter((d) => d.order === 0)];

    const spec = {
        $schema: "https://vega.github.io/schema/vega-lite/v6.json",
        autosize: { contains: "padding" as const, type: "fit" as const },
        background: "transparent",
        config: {
            legend: {
                labelColor: "oklch(var(--foreground))",
                titleColor: "oklch(var(--foreground))"
            },
            view: { stroke: "transparent" }
        },
        data: { values: closedData },
        encoding: {
            color: {
                field: "series",
                legend: {
                    orient: "bottom" as const
                },
                scale: {
                    domain: [groupLabel, averageLabel ?? "Average"],
                    range: [config.group.color, config.average.color]
                },
                type: "nominal" as const
            },
            radius: {
                field: "value",
                scale: {
                    type: "linear" as const,
                    zero: true
                },
                type: "quantitative" as const
            },
            theta: {
                field: "order",
                scale: {
                    domain: [0, labels.length]
                },
                type: "quantitative" as const
            },
            tooltip: [
                {
                    field: "category",
                    title: "Category",
                    type: "nominal" as const
                },
                { field: "series", title: "Series", type: "nominal" as const },
                {
                    field: "value",
                    title: "Value",
                    type: "quantitative" as const
                }
            ]
        },
        mark: {
            point: true,
            type: "line" as const
        },
        width: "container" as const
    };

    return (
        <ChartContainer config={config} className="h-full w-full">
            <VegaEmbed spec={spec} options={{ actions: false }} />
        </ChartContainer>
    );
};
