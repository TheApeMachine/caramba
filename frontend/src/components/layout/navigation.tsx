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
import { useTranslation } from "react-i18next";
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
	const { t } = useTranslation();

	return (
		<Accordion className="w-full">
			<AccordionItem value="item-1">
				<AccordionTrigger>
					<BlocksIcon /> {t("nav.projects")}
				</AccordionTrigger>
				<AccordionPanel className="flex flex-col gap-2">
					<Link to={"/kanban"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<KanbanIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>{t("nav.kanban.title")}</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									{t("nav.kanban.description")}
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
								<h3>{t("nav.requestFeature.title")}</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									{t("nav.requestFeature.description")}
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
					<MicroscopeIcon /> {t("nav.research")}
				</AccordionTrigger>
				<AccordionPanel className="flex flex-col gap-2">
					<Link to={"/nodegraph"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<NetworkIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>{t("nav.architecture.title")}</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									{t("nav.architecture.description")}
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
								<h3>{t("nav.benchmarks.title")}</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									{t("nav.benchmarks.description")}
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
								<h3>{t("nav.newBenchmark.title")}</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									{t("nav.newBenchmark.description")}
								</p>
							</Flex.Column>
							<ChevronRightIcon
								aria-hidden="true"
								className="in-[[data-slot=button]:hover]:translate-x-0.5 transition-transform"
							/>
						</Button>
					</Link>
					<Link to={"/research/new"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<MicroscopeIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>{t("nav.newResearchProject.title")}</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									{t("nav.newResearchProject.description")}
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
					<NetworkIcon /> {t("nav.models")}
				</AccordionTrigger>
				<AccordionPanel className="flex flex-col gap-2">
					<Link to={"/nodegraph"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<CpuIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>{t("nav.architecture.title")}</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									{t("nav.architecture.description")}
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
					<BotIcon /> {t("nav.agents")}
				</AccordionTrigger>
				<AccordionPanel className="flex flex-col gap-2">
					<Link to={"/nodegraph"} onClick={onNavigate}>
						<Button
							className="w-full h-auto! flex flex-row items-center justify-between gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<BotIcon className="shrink-0" />
							<Flex.Column gap={1} className="text-left" fullWidth>
								<h3>{t("nav.architecture.title")}</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									{t("nav.architecture.description")}
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
