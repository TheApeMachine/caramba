/**
 * Grid Examples
 * Demonstrates the different grid variants and their responsive behavior
 */

import { Grid } from "./grid";

interface ExampleCardProps {
    title: string;
    number: number;
}

const ExampleCard = ({ title, number }: ExampleCardProps) => (
    <div className="bg-surface-2 rounded-lg p-6 border border-surface-3 min-h-[120px] flex items-center justify-center">
        <div className="text-center">
            <div className="text-2xl font-bold">{number}</div>
            <div className="text-sm text-muted-foreground mt-1">{title}</div>
        </div>
    </div>
);

// Example 1: Standard responsive grid
export const GridResponsiveExample = () => {
    const items = Array.from({ length: 6 }, (_, i) => ({
        id: i + 1,
        title: "Card"
    }));

    return (
        <div className="space-y-4">
            <div>
                <h3 className="text-lg font-semibold mb-2">
                    Grid.Responsive (cols=6)
                </h3>
                <p className="text-sm text-muted-foreground mb-4">
                    Responsive breakpoints: 1 col mobile → 2 col sm → 3 col md →
                    4 col lg → 6 col xl
                </p>
            </div>
            <Grid cols={6} gap={4}>
                {items.map((item) => (
                    <ExampleCard
                        key={item.id}
                        title={item.title}
                        number={item.id}
                    />
                ))}
            </Grid>
        </div>
    );
};

// Example 2: Auto-fit grid
export const GridAutoExample = () => {
    const items = Array.from({ length: 8 }, (_, i) => ({
        id: i + 1,
        title: "Card"
    }));

    return (
        <div className="space-y-4">
            <div>
                <h3 className="text-lg font-semibold mb-2">
                    Grid.Auto (minWidth=250px)
                </h3>
                <p className="text-sm text-muted-foreground mb-4">
                    Automatically fits items based on container width, minimum
                    250px per item
                </p>
            </div>
            <Grid.Auto minWidth="250px" gap={4}>
                {items.map((item) => (
                    <ExampleCard
                        key={item.id}
                        title={item.title}
                        number={item.id}
                    />
                ))}
            </Grid.Auto>
        </div>
    );
};

// Example 3: Adaptive grid (quantity queries)
export const GridAdaptiveExample = () => {
    const examples = [
        { count: 3, description: "3 items: 2 cols mobile, 3 cols desktop" },
        {
            count: 5,
            description: "5 items: Optimized layout with featured first"
        },
        { count: 6, description: "6 items: 2×3 mobile, 3×2 desktop" },
        { count: 9, description: "9 items: 3×3 grid" }
    ];

    return (
        <div className="space-y-8">
            <div>
                <h3 className="text-lg font-semibold mb-2">
                    Grid.Adaptive (Quantity Queries)
                </h3>
                <p className="text-sm text-muted-foreground mb-4">
                    Layout automatically optimizes based on the exact number of
                    items
                </p>
            </div>

            {examples.map(({ count, description }) => (
                <div key={count} className="space-y-2">
                    <p className="text-sm font-medium">{description}</p>
                    <Grid.Adaptive gap={4}>
                        {Array.from({ length: count }, (_, i) => {
                            const id = `${count}-${i + 1}`;
                            return (
                                <ExampleCard
                                    key={id}
                                    title="Card"
                                    number={i + 1}
                                />
                            );
                        })}
                    </Grid.Adaptive>
                </div>
            ))}
        </div>
    );
};

// Example 4: Comparison - same content, different approaches
export const GridComparisonExample = () => {
    const items = Array.from({ length: 7 }, (_, i) => ({
        id: i + 1,
        title: "Card"
    }));

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-xl font-bold mb-4">
                    Comparison: 7 Items in Different Grids
                </h2>
            </div>

            <div className="space-y-2">
                <h3 className="font-semibold">Standard Grid (cols=3)</h3>
                <p className="text-sm text-muted-foreground">
                    Fixed 3 columns at desktop, last row not filled
                </p>
                <Grid cols={3} gap={4}>
                    {items.map((item) => (
                        <ExampleCard
                            key={item.id}
                            title={item.title}
                            number={item.id}
                        />
                    ))}
                </Grid>
            </div>

            <div className="space-y-2">
                <h3 className="font-semibold">Grid.Auto (minWidth=250px)</h3>
                <p className="text-sm text-muted-foreground">
                    Items auto-fit, may have uneven rows
                </p>
                <Grid.Auto minWidth="250px" gap={4}>
                    {items.map((item) => (
                        <ExampleCard
                            key={item.id}
                            title={item.title}
                            number={item.id}
                        />
                    ))}
                </Grid.Auto>
            </div>

            <div className="space-y-2">
                <h3 className="font-semibold">Grid.Adaptive</h3>
                <p className="text-sm text-muted-foreground">
                    First item featured (2×2), rest in optimal grid ✨
                </p>
                <Grid.Adaptive gap={4}>
                    {items.map((item) => (
                        <ExampleCard
                            key={item.id}
                            title={item.title}
                            number={item.id}
                        />
                    ))}
                </Grid.Adaptive>
            </div>
        </div>
    );
};

// Example 5: Using with FFComponent
export const GridWithFFComponentExample = () => {
    return (
        <div className="space-y-4">
            <div>
                <h3 className="text-lg font-semibold mb-2">
                    Grid + FFComponent (Dashboard Layout)
                </h3>
                <p className="text-sm text-muted-foreground mb-4">
                    FFComponent cols are now responsive and adapt to screen size
                </p>
            </div>
            <Grid cols={6} gap={6}>
                {/* These will be 1 col on mobile, 2 on tablet, 3 on desktop */}
                <div className="col-span-1 md:col-span-2 lg:col-span-3">
                    <ExampleCard title="Spans 3 cols" number={1} />
                </div>
                <div className="col-span-1 md:col-span-2 lg:col-span-3">
                    <ExampleCard title="Spans 3 cols" number={2} />
                </div>
                <div className="col-span-1 md:col-span-2 lg:col-span-2">
                    <ExampleCard title="Spans 2 cols" number={3} />
                </div>
                <div className="col-span-1 md:col-span-2 lg:col-span-2">
                    <ExampleCard title="Spans 2 cols" number={4} />
                </div>
                <div className="col-span-1 md:col-span-2 lg:col-span-2">
                    <ExampleCard title="Spans 2 cols" number={5} />
                </div>
            </Grid>
        </div>
    );
};
