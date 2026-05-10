"use client";

import { useMemo } from "react";
import { VegaEmbed } from "react-vega";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle
} from "@/components/ui/card";
import {
    type ChartConfig,
    ChartContainer,
    ChartStyle
} from "@/components/ui/chart";

export const description = "An interactive pie chart";

const desktopData = [
    { desktop: 186, fill: "var(--color-january)", month: "january" },
    { desktop: 305, fill: "var(--color-february)", month: "february" },
    { desktop: 237, fill: "var(--color-march)", month: "march" },
    { desktop: 173, fill: "var(--color-april)", month: "april" },
    { desktop: 209, fill: "var(--color-may)", month: "may" }
];

const chartConfig = {
    april: {
        color: "oklch(var(--chart-4))",
        label: ""
    },
    desktop: {
        label: "Desktop"
    },
    february: {
        color: "oklch(var(--chart-2))",
        label: "February"
    },
    january: {
        color: "oklch(var(--chart-1))",
        label: "January"
    },
    march: {
        color: "oklch(var(--chart-3))",
        label: "March"
    },
    may: {
        color: "oklch(var(--chart-5))",
        label: "May"
    },
    mobile: {
        label: "Mobile"
    },
    visitors: {
        label: "Visitors"
    }
} satisfies ChartConfig;

export function ChartDonut() {
    const id = "pie-interactive";
    const activeIndex = 0;

    const activeValue = desktopData[activeIndex].desktop;

    const spec = useMemo(
        () => ({
            $schema: "https://vega.github.io/schema/vega-lite/v6.json",
            autosize: { contains: "padding" as const, type: "fit" as const },
            background: "transparent",
            config: {
                view: { stroke: null }
            },
            data: {
                values: desktopData.map((row) => ({
                    color: row.fill,
                    month: row.month,
                    value: row.desktop
                }))
            },
            encoding: {
                color: {
                    field: "month",
                    legend: null,
                    scale: {
                        domain: desktopData.map((d) => d.month),
                        range: desktopData.map((d) => d.fill)
                    },
                    type: "nominal" as const
                },
                theta: {
                    field: "value",
                    type: "quantitative" as const
                },
                tooltip: [
                    {
                        field: "month",
                        title: "Month",
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
                innerRadius: 100,
                outerRadius: 150,
                type: "arc" as const
            },
            width: "container" as const
        }),
        []
    );

    return (
        <Card data-chart={id} className="flex flex-col">
            <ChartStyle id={id} config={chartConfig} />
            <CardHeader className="flex-row items-start space-y-0 pb-0">
                <div className="grid gap-1">
                    <CardTitle>Pie Chart - Interactive</CardTitle>
                    <CardDescription>January - June 2024</CardDescription>
                </div>
            </CardHeader>
            <CardContent className="flex flex-1 justify-center pb-0">
                <ChartContainer
                    id={id}
                    config={chartConfig}
                    className="mx-auto aspect-square w-full max-w-[300px] relative"
                >
                    <VegaEmbed spec={spec} options={{ actions: false }} />
                    <div
                        style={{
                            alignItems: "center",
                            display: "flex",
                            flexDirection: "column",
                            inset: 0,
                            justifyContent: "center",
                            pointerEvents: "none",
                            position: "absolute"
                        }}
                    >
                        <div style={{ fontSize: 28, fontWeight: 700 }}>
                            {activeValue.toLocaleString()}
                        </div>
                        <div style={{ fontSize: 12, opacity: 0.7 }}>
                            Visitors
                        </div>
                    </div>
                </ChartContainer>
            </CardContent>
        </Card>
    );
}

export default ChartDonut;
