"use client";

import { useMemo, useState } from "react";
import { VegaEmbed } from "react-vega";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle
} from "@/components/ui/card";
import { type ChartConfig, ChartContainer } from "@/components/ui/chart";
import { getColorIndex } from "./color-utils";

type SeriesPoint = { date: number } & Record<string, number>;

interface ChartAreaInteractiveProps {
    title?: string;
    description?: string;
    data: SeriesPoint[];
    config: ChartConfig;
    disableInternalFiltering?: boolean;
}

export function ChartAreaInteractive({
    title,
    description,
    data,
    config,
    disableInternalFiltering = false
}: ChartAreaInteractiveProps) {
    const [timeRange, _setTimeRange] = useState("90d");

    const filteredData = useMemo(
        () =>
            disableInternalFiltering
                ? data
                : data.filter((item) => {
                      const date = new Date(item.date);
                      const referenceDate = new Date();
                      let daysToSubtract = 90;
                      if (timeRange === "30d") {
                          daysToSubtract = 30;
                      } else if (timeRange === "7d") {
                          daysToSubtract = 7;
                      }
                      const startDate = new Date(referenceDate);
                      startDate.setDate(startDate.getDate() - daysToSubtract);
                      return date.getTime() >= startDate.getTime();
                  }),
        [data, disableInternalFiltering, timeRange]
    );

    const seriesKeys = useMemo(() => Object.keys(config), [config]);

    const finalConfig: ChartConfig = useMemo(
        () =>
            seriesKeys.reduce<ChartConfig>((acc, key, idx) => {
                const current = config[key] ?? {};
                const hasExplicitColor = Object.hasOwn(current, "color");
                const hasTheme = Object.hasOwn(current, "theme");
                const colorIndex = (idx % 10) + 1;
                acc[key] = {
                    icon: current.icon,
                    label: current.label,
                    ...(hasTheme || hasExplicitColor
                        ? current
                        : { color: `oklch(var(--chart-${colorIndex}))` })
                };
                return acc;
            }, {}),
        [seriesKeys, config]
    );

    // Transform data for Vega-Lite
    const vegaData = useMemo(() => {
        return filteredData.flatMap((item) =>
            seriesKeys.map((key) => ({
                color:
                    finalConfig[key]?.color ||
                    `oklch(var(--chart-${getColorIndex(seriesKeys.indexOf(key))}))`,
                date: item.date,
                series:
                    typeof finalConfig[key]?.label === "string"
                        ? finalConfig[key].label
                        : key,
                value: item[key] ?? 0
            }))
        );
    }, [filteredData, seriesKeys, finalConfig]);

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
        data: { values: vegaData },
        encoding: {
            color: {
                field: "series",
                legend: null,
                scale: {
                    domain: seriesKeys.map((key) =>
                        typeof finalConfig[key]?.label === "string"
                            ? finalConfig[key].label
                            : key
                    ),
                    range: seriesKeys.map(
                        (key) =>
                            finalConfig[key]?.color ||
                            `oklch(var(--chart-${getColorIndex(seriesKeys.indexOf(key))}))`
                    )
                },
                type: "nominal" as const
            },
            tooltip: [
                {
                    field: "date",
                    format: "%b %d, %Y",
                    title: "Date",
                    type: "temporal" as const
                },
                { field: "series", title: "Series", type: "nominal" as const },
                {
                    field: "value",
                    title: "Value",
                    type: "quantitative" as const
                }
            ],
            x: {
                axis: {
                    format: "%b %d",
                    grid: false,
                    title: null
                },
                field: "date",
                type: "temporal" as const
            },
            y: {
                axis: {
                    grid: false,
                    title: null
                },
                field: "value",
                scale: { domain: [0, null] },
                type: "quantitative" as const
            }
        },
        mark: {
            line: true,
            opacity: 0.3,
            point: false,
            type: "area" as const
        },
        width: "container" as const
    };

    return (
        <Card className="w-full">
            {title && (
                <CardHeader>
                    <CardTitle>{title}</CardTitle>
                    <CardDescription>{description}</CardDescription>
                </CardHeader>
            )}
            <CardContent className="overflow-visible p-0">
                <ChartContainer config={finalConfig} className="w-full h-full">
                    <VegaEmbed spec={spec} options={{ actions: false }} />
                </ChartContainer>
            </CardContent>
        </Card>
    );
}
