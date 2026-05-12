"use client";

import { VegaEmbed } from "react-vega";
import { Flex } from "../ui/flex";

interface GaugeProps {
    value?: number;
    maxValue?: number;
    precision?: number;
    suffix?: string;
    width?: number;
}

export const Gauge = ({
    value,
    maxValue,
    precision,
    suffix = "",
    width = 400
}: GaugeProps) => {
    const normalizedMax =
        typeof maxValue === "number" &&
        Number.isFinite(maxValue) &&
        maxValue > 0
            ? maxValue
            : 100;

    const normalizedValue =
        typeof value === "number" && Number.isFinite(value) ? value : 0;

    const displayPrecision =
        typeof precision === "number" && Number.isFinite(precision)
            ? precision
            : 0;

    // Scale factor based on desired width vs design width (400)
    const scale = width / 400;

    // Scaled dimensions
    const outerRadius = 160 * scale;
    const innerRadius = 110 * scale;
    const needleLength = 110 * scale;
    const tickLabelRadius = 185 * scale;
    const heightOffset = 11 * scale;
    const centerValueOffset = 61 * scale;

    const ticks = Array.from({ length: 6 }, (_, i) => {
        const val = (normalizedMax / 5) * i;
        return {
            label: Math.round(val).toString(),
            value: val
        };
    });

    // Create gauge spec using Vega (not Vega-Lite) for arc marks
    const spec = {
        $schema: "https://vega.github.io/schema/vega/v6.json" as const,
        autosize: { contains: "padding" as const, type: "fit" as const },
        background: "transparent",
        data: [
            {
                name: "table",
                transform: [
                    {
                        endAngle: { signal: "PI * 0.55" },
                        field: "value",
                        startAngle: { signal: "-PI * 0.55" },
                        type: "pie" as const
                    }
                ],
                values: [
                    {
                        category: 1,
                        value: normalizedMax * 0.5
                    },
                    {
                        category: 2,
                        value: normalizedMax * 0.25
                    },
                    {
                        category: 3,
                        value: normalizedMax * 0.25
                    }
                ]
            },
            {
                name: "needle",
                transform: [
                    {
                        as: "angle",
                        expr: `(-PI * 0.55) + ((datum.value / ${normalizedMax}) * PI * 1.1)`,
                        type: "formula" as const
                    },
                    {
                        as: "x",
                        expr: `(width / 2) + ${needleLength} * sin(datum.angle)`,
                        type: "formula" as const
                    },
                    {
                        as: "y",
                        expr: `(height - ${heightOffset}) - ${needleLength} * cos(datum.angle)`,
                        type: "formula" as const
                    }
                ],
                values: [{ value: normalizedValue }]
            },
            {
                name: "ticks",
                transform: [
                    {
                        as: "angle",
                        expr: `(-PI * 0.55) + ((datum.value / ${normalizedMax}) * PI * 1.1)`,
                        type: "formula" as const
                    }
                ],
                values: ticks
            }
        ],
        height: 280 * scale,
        marks: [
            // Background gauge segments (dimmed)
            {
                encode: {
                    enter: {
                        cornerRadius: { value: 3 * scale },
                        endAngle: { field: "endAngle" },
                        fill: { field: "category", scale: "gaugeColor" },
                        innerRadius: { value: innerRadius },
                        opacity: { value: 0.2 },
                        outerRadius: { value: outerRadius },
                        padAngle: { value: 0.02 },
                        startAngle: { field: "startAngle" },
                        x: { signal: "width / 2" },
                        y: { signal: `height - ${heightOffset}` }
                    }
                },
                from: { data: "table" },
                type: "arc" as const
            },
            // Foreground gauge segments (filled to value)
            {
                encode: {
                    enter: {
                        cornerRadius: { value: 3 * scale },
                        endAngle: {
                            signal: `min(datum.endAngle, (-PI * 0.55) + ((${normalizedValue} / ${normalizedMax}) * PI * 1.1))`
                        },
                        fill: { field: "category", scale: "gaugeColor" },
                        innerRadius: { value: innerRadius },
                        outerRadius: { value: outerRadius },
                        padAngle: { value: 0.02 },
                        startAngle: { field: "startAngle" },
                        x: { signal: "width / 2" },
                        y: { signal: `height - ${heightOffset}` }
                    }
                },
                from: { data: "table" },
                type: "arc" as const
            },
            // Triangular pointer shadow
            {
                encode: {
                    enter: {
                        fill: { value: "#000000" },
                        opacity: { value: 0.3 },
                        shape: {
                            value: "M 0,-4.33 L 2.5,0 L -2.5,0 Z"
                        },
                        size: { value: 100 * scale },
                        x: { field: "x", offset: { value: 1 * scale } },
                        y: { field: "y", offset: { value: 2 * scale } }
                    },
                    update: {
                        angle: { signal: "datum.angle * 180 / PI" }
                    }
                },
                from: { data: "needle" },
                type: "symbol" as const
            },
            // Triangular pointer
            {
                encode: {
                    enter: {
                        fill: { value: "#ffffff" },
                        shape: {
                            value: "M 0,-4.33 L 2.5,0 L -2.5,0 Z"
                        },
                        size: { value: 100 * scale },
                        stroke: { value: "#e5e7eb" },
                        strokeWidth: { value: 0.5 },
                        x: { field: "x" },
                        y: { field: "y" }
                    },
                    update: {
                        angle: { signal: "datum.angle * 180 / PI" }
                    }
                },
                from: { data: "needle" },
                type: "symbol" as const
            },
            // Tick labels (outside the gauge)
            {
                encode: {
                    enter: {
                        align: { value: "center" as const },
                        baseline: { value: "middle" as const },
                        fill: { value: "#d1d5db" },
                        fontSize: { value: 20 * scale },
                        text: { field: "label" },
                        x: {
                            signal: `(width / 2) + ${tickLabelRadius} * sin(datum.angle)`
                        },
                        y: {
                            signal: `(height - ${heightOffset}) - ${tickLabelRadius} * cos(datum.angle)`
                        }
                    }
                },
                from: { data: "ticks" },
                type: "text" as const
            },
            // Center value display
            {
                encode: {
                    enter: {
                        align: { value: "center" as const },
                        baseline: { value: "top" as const },
                        fill: { value: "#ffffff" },
                        fontSize: { value: 32 * scale },
                        text: {
                            signal: `${displayPrecision > 0 ? `format(${normalizedValue}, '.${displayPrecision}f')` : `format(${normalizedValue}, 'd')`} + '${suffix}'`
                        },
                        x: { signal: "width / 2" },
                        y: { signal: `height - ${centerValueOffset}` }
                    }
                },
                type: "text" as const
            }
        ],
        responsive: true,
        scales: [
            {
                domain: [1, 2, 3],
                name: "gaugeColor",
                range: [
                    "var(--color-gauge-1)",
                    "var(--color-gauge-3)",
                    "var(--color-gauge-4)"
                ],
                type: "ordinal" as const
            }
        ],
        width: width
    };

    return (
        <Flex.Column
            style={{
                background: "transparent"
            }}
            data-ai-selector="gauge"
            data-ai-value={value}
        >
            <VegaEmbed spec={spec} options={{ actions: false }} />
        </Flex.Column>
    );
};
