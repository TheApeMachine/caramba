"use client";

import { FolderIcon, PlusIcon, X } from "lucide-react";
import { type ReactNode, useMemo } from "react";
import { Button } from "#/components/ui/button";
import { Empty } from "#/components/ui/empty";
import {
	Card,
	CardFrame,
	CardFrameAction,
	CardFrameDescription,
	CardFrameHeader,
	CardFrameTitle,
	CardPanel,
} from "@/components/ui/card";
import type { WidgetDescriptor } from "./registry";

/*
Widget renders one descriptor inside the dashboard's card chrome. A
descriptor either supplies a Vega spec (chart path) or a React node
(arbitrary content). Drag/resize behavior is layered on by the caller;
`overlay` is rendered inside the same hover group so handles can use
group-hover transitions.
*/
interface WidgetProps {
	title: string;
	description: string;
	buttons: ReactNode[];
	onRemove?: () => void;
}

export const Widget = ({
	title,
	description,
	buttons,
	onRemove,
}: WidgetProps) => {
	return (
		<CardFrame className="w-full">
			<CardFrameHeader>
				<CardFrameTitle>{title}</CardFrameTitle>
				<CardFrameDescription>{description}</CardFrameDescription>
				<CardFrameAction>{buttons}</CardFrameAction>
			</CardFrameHeader>
			<Card className="h-full">
				<CardPanel>
					<Empty>
						<Empty.Header>
							<Empty.Media variant="icon">
								<FolderIcon />
							</Empty.Media>
							<Empty.Title>No projects yet</Empty.Title>
							<Empty.Description>
								Get started by adding your first project.
							</Empty.Description>
						</Empty.Header>
					</Empty>
				</CardPanel>
			</Card>
		</CardFrame>
	);
};
