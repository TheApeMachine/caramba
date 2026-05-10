"use client";

import { useTranslation } from "react-i18next";
import { VegaEmbed } from "react-vega";

interface ProgressChartProps {
    data?: { finished: number; notStarted: number; started: number };
}

export const ProgressChart = ({
    data = { finished: 60, notStarted: 25, started: 15 }
}: ProgressChartProps) => {
    const { t } = useTranslation();

    const labels = [
        t("measurement.card.progress.notStarted"),
        t("measurement.card.progress.started"),
        t("measurement.card.progress.finished")
    ];

    const GRADIENT_COLORS = ["#ef4444", "#fb923c", "#4ade80"];

    // Calculate cumulative positions
    const segments = [
        {
            color: GRADIENT_COLORS[0],
            end: data.notStarted,
            label: labels[0],
            start: 0,
            value: data.notStarted
        },
        {
            color: GRADIENT_COLORS[1],
            end: data.notStarted + data.started,
            label: labels[1],
            start: data.notStarted,
            value: data.started
        },
        {
            color: GRADIENT_COLORS[2],
            end: data.notStarted + data.started + data.finished,
            label: labels[2],
            start: data.notStarted + data.started,
            value: data.finished
        }
    ];

    const total = data.notStarted + data.started + data.finished;

    const spec = {
        $schema: "https://vega.github.io/schema/vega-lite/v6.json",
        autosize: { contains: "padding" as const, type: "fit" as const },
        background: "transparent",
        config: {
            view: { stroke: "transparent" }
        },
        data: { values: segments },
        layer: [
            {
                encoding: {
                    color: {
                        field: "color",
                        legend: null,
                        scale: null,
                        type: "nominal" as const
                    },
                    x: {
                        axis: null,
                        field: "start",
                        scale: { domain: [0, total] },
                        type: "quantitative" as const
                    },
                    x2: { field: "end" }
                },
                mark: {
                    type: "bar" as const
                }
            },
            {
                encoding: {
                    color: { value: "oklch(var(--foreground))" },
                    text: { field: "label", type: "nominal" as const },
                    x: {
                        field: "midpoint",
                        type: "quantitative" as const
                    }
                },
                mark: {
                    align: "center" as const,
                    baseline: "middle" as const,
                    fontSize: 11,
                    type: "text" as const
                },
                transform: [
                    {
                        as: "midpoint",
                        calculate: "(datum.start + datum.end) / 2"
                    }
                ]
            }
        ],
        width: "container" as const
    };

    return (
        <div
            style={{
                background: "transparent",
                height: 50,
                minHeight: 50,
                width: "100%"
            }}
        >
            <VegaEmbed spec={spec} options={{ actions: false }} />
        </div>
    );
};
