import { Link } from "@tanstack/react-router";
import {
	BlocksIcon,
	BotIcon,
	ChevronRightIcon,
	CpuIcon,
	FlaskConicalIcon,
	GaugeIcon,
	KanbanIcon,
	LightbulbIcon,
	MicroscopeIcon,
	NetworkIcon,
} from "lucide-react";
import {
	Accordion,
	AccordionItem,
	AccordionPanel,
	AccordionTrigger,
} from "#/components/ui/accordion";
import { Button } from "#/components/ui/button";
import { Flex } from "../ui/flex";

export const Navigation = ({
	onNavigate,
}: {
	onNavigate?: () => void;
} = {}) => {
	return (
		<Accordion className="w-full">
			<AccordionItem value="item-1">
				<AccordionTrigger>
					<BlocksIcon /> Projects
				</AccordionTrigger>
				<AccordionPanel className="flex flex-col gap-2">
					<Link to={"/kanban"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<KanbanIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>Kanban</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									Manage your projects
								</p>
							</Flex.Column>
							<ChevronRightIcon
								aria-hidden="true"
								className="in-[[data-slot=button]:hover]:translate-x-0.5 transition-transform"
							/>
						</Button>
					</Link>
					<Link to={"/request-feature"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<LightbulbIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>Request a feature</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									Send requests to the Requests backlog
								</p>
							</Flex.Column>
							<ChevronRightIcon
								aria-hidden="true"
								className="in-[[data-slot=button]:hover]:translate-x-0.5 transition-transform"
							/>
						</Button>
					</Link>
				</AccordionPanel>
			</AccordionItem>
			<AccordionItem value="item-2">
				<AccordionTrigger>
					<MicroscopeIcon /> Research
				</AccordionTrigger>
				<AccordionPanel className="flex flex-col gap-2">
					<Link to={"/research"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<MicroscopeIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>Architecture</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									Build and manage your research architectures
								</p>
							</Flex.Column>
							<ChevronRightIcon
								aria-hidden="true"
								className="in-[[data-slot=button]:hover]:translate-x-0.5 transition-transform"
							/>
						</Button>
					</Link>
					<Link to={"/benchmarks"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<GaugeIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>Benchmarks</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									Run evaluations and watch them live
								</p>
							</Flex.Column>
							<ChevronRightIcon
								aria-hidden="true"
								className="in-[[data-slot=button]:hover]:translate-x-0.5 transition-transform"
							/>
						</Button>
					</Link>
					<Link to={"/benchmarks/new"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<FlaskConicalIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>New benchmark</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									Pick a preset or configure from scratch
								</p>
							</Flex.Column>
							<ChevronRightIcon
								aria-hidden="true"
								className="in-[[data-slot=button]:hover]:translate-x-0.5 transition-transform"
							/>
						</Button>
					</Link>
				</AccordionPanel>
			</AccordionItem>
			<AccordionItem value="item-3">
				<AccordionTrigger>
					<NetworkIcon /> Models
				</AccordionTrigger>
				<AccordionPanel className="flex flex-col gap-2">
					<Link to={"/research"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<CpuIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>Architecture</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									Build and manage your research architectures
								</p>
							</Flex.Column>
							<ChevronRightIcon
								aria-hidden="true"
								className="in-[[data-slot=button]:hover]:translate-x-0.5 transition-transform"
							/>
						</Button>
					</Link>
				</AccordionPanel>
			</AccordionItem>
			<AccordionItem value="item-4">
				<AccordionTrigger>
					<BotIcon /> Agents
				</AccordionTrigger>
				<AccordionPanel className="flex flex-col gap-2">
					<Link to={"/research"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<BotIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>Architecture</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									Build and manage your research architectures
								</p>
							</Flex.Column>
							<ChevronRightIcon
								aria-hidden="true"
								className="in-[[data-slot=button]:hover]:translate-x-0.5 transition-transform"
							/>
						</Button>
					</Link>
				</AccordionPanel>
			</AccordionItem>
		</Accordion>
	);
};
