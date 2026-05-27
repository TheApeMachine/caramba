import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import {
	DynamicIslandPreset,
	SHAPE_ORDER,
	SHAPES,
	type ShapeKey,
} from "#/components/ui/dynamic-island";
import { Flex } from "#/components/ui/flex";
import { Button } from "@/components/ui/button";

export const DynamicIslandPlayground = () => {
	const [shapeKey, setShapeKey] = useState<ShapeKey>("button");

	return (
		<Flex.Column
			className="min-h-[640px] bg-background text-foreground"
			fullHeight
		>
			<Flex.Center className="flex-1">
				<DynamicIslandPreset shapeKey={shapeKey} />
			</Flex.Center>

			<Flex.Row
				className="flex-wrap justify-center gap-2 px-5 pb-7"
				role="toolbar"
				aria-label="Island shape presets"
			>
				{SHAPE_ORDER.map((nextShapeKey) => (
					<Button
						key={nextShapeKey}
						type="button"
						variant={nextShapeKey === shapeKey ? "brand" : "secondary"}
						size="sm"
						aria-pressed={nextShapeKey === shapeKey}
						onClick={() => setShapeKey(nextShapeKey)}
					>
						{SHAPES[nextShapeKey].label}
					</Button>
				))}
			</Flex.Row>
		</Flex.Column>
	);
};

export const Route = createFileRoute("/test")({
	component: DynamicIslandPlayground,
});
