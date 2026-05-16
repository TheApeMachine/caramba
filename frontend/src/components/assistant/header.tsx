import { Bot, Maximize2, Minimize2, X } from "lucide-react";
import { Button } from "#/components/ui/button";
import {
	CardFrameAction,
	CardFrameDescription,
	CardFrameHeader,
	CardFrameTitle,
} from "#/components/ui/card";
import { Flex } from "#/components/ui/flex";
import { cn } from "#/lib/utils";
import type { Mode } from "./types";

interface HeaderProps {
	mode: Mode;
	setMode: (mode: Mode) => void;
	teamName: string;
}

export const Header = ({ mode, setMode, teamName }: HeaderProps) => {
	const isClosed = mode === "closed";
	const isMini = mode === "mini";
	const isFull = mode === "full";

	return (
		<CardFrameHeader
			className={cn(isClosed && "p-0")}
			onClick={isClosed ? () => setMode("mini") : undefined}
		>
			<Flex.Row align="center" gap={2}>
				<Flex
					align="center"
					justify="center"
					className={cn(
						"shrink-0",
						isClosed
							? "size-14 cursor-pointer"
							: "size-7 rounded-md bg-primary text-primary-foreground",
					)}
				>
					<Bot className={isClosed ? "size-6" : "size-4"} />
				</Flex>
				{!isClosed && <CardFrameTitle>{teamName}</CardFrameTitle>}
			</Flex.Row>
			{isFull && (
				<CardFrameDescription>
					Manage sessions, personas, and chat with your team.
				</CardFrameDescription>
			)}
			{!isClosed && (
				<CardFrameAction>
					<Flex.Row align="center" gap={1}>
						<Button
							size="icon-xs"
							variant="ghost"
							onClick={() => setMode(isMini ? "full" : "mini")}
							aria-label={isMini ? "Expand" : "Collapse"}
						>
							{isMini ? <Maximize2 /> : <Minimize2 />}
						</Button>
						<Button
							size="icon-xs"
							variant="ghost"
							onClick={() => setMode("closed")}
							aria-label="Close"
						>
							<X />
						</Button>
					</Flex.Row>
				</CardFrameAction>
			)}
		</CardFrameHeader>
	);
};
