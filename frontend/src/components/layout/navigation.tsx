import { Link } from "@tanstack/react-router";
import {
	BotIcon,
	ChevronRightIcon,
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

export const Navigation = () => {
	return (
		<Accordion className="w-full">
			<AccordionItem value="item-1">
				<AccordionTrigger>
					<MicroscopeIcon /> Research
				</AccordionTrigger>
				<AccordionPanel>
					<Link to={"/research"}>
						<Button
							className="h-auto! gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<div className="flex flex-col gap-0.5">
								<h3>Architecture</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									Build and manage your research architectures
								</p>
							</div>
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
					<NetworkIcon /> Models
				</AccordionTrigger>
				<AccordionPanel>
					<Link to={"/research"}>
						<Button
							className="h-auto! gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<div className="flex flex-col gap-0.5">
								<h3>Architecture</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									Build and manage your research architectures
								</p>
							</div>
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
					<BotIcon /> Agents
				</AccordionTrigger>
				<AccordionPanel>
					<Link to={"/research"}>
						<Button
							className="h-auto! gap-4 px-4 py-3 text-left"
							variant="outline"
						>
							<div className="flex flex-col gap-0.5">
								<h3>Architecture</h3>
								<p className="whitespace-break-spaces font-normal text-muted-foreground">
									Build and manage your research architectures
								</p>
							</div>
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
