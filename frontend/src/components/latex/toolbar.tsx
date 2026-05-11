import {
	AlignCenterIcon,
	AlignRightIcon,
	DollarSignIcon,
	InfoIcon,
	PercentIcon,
	TableOfContentsIcon,
} from "lucide-react";
import { Button } from "#/components/ui/button";
import { Form } from "#/components/ui/form";
import {
	Sheet,
	SheetClose,
	SheetDescription,
	SheetFooter,
	SheetHeader,
	SheetPanel,
	SheetPopup,
	SheetTitle,
	SheetTrigger,
} from "#/components/ui/sheet";
import { ToggleGroup, ToggleGroupItem } from "#/components/ui/toggle-group";
import {
	Toolbar,
	ToolbarButton,
	ToolbarGroup,
	ToolbarSeparator,
} from "#/components/ui/toolbar";
import {
	Tooltip,
	TooltipPopup,
	TooltipProvider,
	TooltipTrigger,
} from "#/components/ui/tooltip";

import { MetadataTab, usePaperMetadataForm } from "./panels/metadata-tab";
import { OutlinePanel } from "./panels/outline-panel";

export const LatexToolbar = () => {
	return (
		<TooltipProvider>
			<Toolbar>
				<ToggleGroup className="border-none p-0" defaultValue={["left"]}>
					<Tooltip>
						<TooltipTrigger
							render={
								<ToolbarButton
									aria-label="Align left"
									render={<ToggleGroupItem value="left" />}
								>
									<Sheet>
										<SheetTrigger render={<Button variant="outline" />}>
											<TableOfContentsIcon />
										</SheetTrigger>
										<SheetPopup variant="inset" side="left">
											<SheetHeader>
												<SheetTitle>Edit profile</SheetTitle>
												<SheetDescription>
													Make changes to your profile here. Click save when
													you&apos;re done.
												</SheetDescription>
											</SheetHeader>
											<Form className="contents">
												<SheetPanel className="grid gap-4">
													<OutlinePanel />
												</SheetPanel>
												<SheetFooter>
													<SheetClose render={<Button variant="ghost" />}>
														Cancel
													</SheetClose>
													<Button type="submit">Save</Button>
												</SheetFooter>
											</Form>
										</SheetPopup>
									</Sheet>
								</ToolbarButton>
							}
						/>
						<TooltipPopup sideOffset={8}>Align left</TooltipPopup>
					</Tooltip>
					<Tooltip>
						<TooltipTrigger
							render={
								<ToolbarButton
									aria-label="Align center"
									render={
										<ToggleGroupItem
											aria-label="Toggle center"
											value="center"
										/>
									}
								>
									<AlignCenterIcon />
								</ToolbarButton>
							}
						/>
						<TooltipPopup sideOffset={8}>Align center</TooltipPopup>
					</Tooltip>
					<Tooltip>
						<TooltipTrigger
							render={
								<ToolbarButton
									aria-label="Align right"
									render={
										<ToggleGroupItem aria-label="Toggle right" value="right" />
									}
								>
									<AlignRightIcon />
								</ToolbarButton>
							}
						/>
						<TooltipPopup sideOffset={8}>Align right</TooltipPopup>
					</Tooltip>
				</ToggleGroup>
				<ToolbarSeparator />
				<ToolbarGroup>
					<Tooltip>
						<TooltipTrigger
							render={
								<ToolbarButton
									aria-label="Format as currency"
									render={<Button size="icon" variant="ghost" />}
								>
									<DollarSignIcon />
								</ToolbarButton>
							}
						/>
						<TooltipPopup sideOffset={8}>Format as currency</TooltipPopup>
					</Tooltip>
					<Tooltip>
						<TooltipTrigger
							render={
								<ToolbarButton
									aria-label="Format as percent"
									render={<Button size="icon" variant="ghost" />}
								>
									<PercentIcon />
								</ToolbarButton>
							}
						/>
						<TooltipPopup sideOffset={8}>Format as percent</TooltipPopup>
					</Tooltip>
				</ToolbarGroup>
				<ToolbarSeparator />
				<ToolbarGroup>
					<ToolbarButton render={<Button />}>
						<Sheet>
							<SheetTrigger render={<Button variant="outline" />}>
								<InfoIcon />
							</SheetTrigger>
							<SheetPopup variant="inset">
								<SheetHeader>
									<SheetTitle>Edit profile</SheetTitle>
									<SheetDescription>
										Make changes to your profile here. Click save when
										you&apos;re done.
									</SheetDescription>
								</SheetHeader>
								<Form>
									<SheetPanel className="grid gap-4">
										<MetadataTab form={usePaperMetadataForm()} />
									</SheetPanel>
									<SheetFooter>
										<SheetClose render={<Button variant="ghost" />}>
											Cancel
										</SheetClose>
										<Button type="submit">Save</Button>
									</SheetFooter>
								</Form>
							</SheetPopup>
						</Sheet>
					</ToolbarButton>
				</ToolbarGroup>
			</Toolbar>
		</TooltipProvider>
	);
};
