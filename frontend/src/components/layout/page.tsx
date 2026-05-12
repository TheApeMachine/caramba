import { Link } from "@tanstack/react-router";
import { FoldersIcon, Menu as MenuIcon } from "lucide-react";
import type React from "react";
import {
	Breadcrumb,
	BreadcrumbItem,
	BreadcrumbLink,
	BreadcrumbList,
	BreadcrumbPage,
	BreadcrumbSeparator,
} from "#/components/ui/breadcrumb";
import { Button } from "#/components/ui/button";
import { Menu, MenuItem, MenuPopup, MenuTrigger } from "#/components/ui/menu";
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
import { Navigation } from "./navigation";

export const Page = ({ children }: { children?: React.ReactNode }) => {
	return <>{children ?? null}</>;
};

Page.Header = ({ children }: { children?: React.ReactNode }) => {
	return (
		<header className="grid-area-header flex shrink-0 flex-wrap items-center justify-between gap-4 p-4">
			<Breadcrumb className="min-w-0 flex-1">
				<BreadcrumbList>
					<BreadcrumbItem>
						<Sheet>
							<SheetTrigger render={<Button variant="outline" />}>
								<MenuIcon />
							</SheetTrigger>
							<SheetPopup variant="inset" side="left">
								<SheetHeader>
									<SheetTitle>Caramba</SheetTitle>
									<SheetDescription>
										A substrate for AI research
									</SheetDescription>
								</SheetHeader>
								<SheetPanel className="grid gap-4">
									<Navigation />
								</SheetPanel>
								<SheetFooter>
									<SheetClose render={<Button variant="ghost" />}>
										Cancel
									</SheetClose>
									<Button type="submit">Save</Button>
								</SheetFooter>
							</SheetPopup>
						</Sheet>
					</BreadcrumbItem>
					<BreadcrumbItem>
						<BreadcrumbLink render={<Link to={"/"} />}>Home</BreadcrumbLink>
					</BreadcrumbItem>
					<BreadcrumbSeparator />
					<BreadcrumbItem>
						<Menu>
							<MenuTrigger
								aria-label="More pages"
								render={
									<Button
										className="-m-1.5 text-muted-foreground"
										size="icon-sm"
										variant="ghost"
									/>
								}
							>
								<FoldersIcon aria-hidden="true" />
							</MenuTrigger>
							<MenuPopup align="start">
								<MenuItem>
									<Link to={"/docs"} />
								</MenuItem>
							</MenuPopup>
						</Menu>
					</BreadcrumbItem>
					<BreadcrumbSeparator />
					<BreadcrumbItem>
						<BreadcrumbLink render={<Link to={"/docs"} />}>
							Components
						</BreadcrumbLink>
					</BreadcrumbItem>
					<BreadcrumbSeparator />
					<BreadcrumbItem>
						<BreadcrumbPage>Breadcrumb</BreadcrumbPage>
					</BreadcrumbItem>
				</BreadcrumbList>
			</Breadcrumb>

			{children ?? null}
		</header>
	);
};

Page.Nav = ({ children }: { children?: React.ReactNode[] }) => {
	return <nav className="grid-area-nav shrink-0">{children ?? null}</nav>;
};

Page.Main = ({ children }: { children?: React.ReactNode }) => {
	return (
		<main className="flex h-full min-h-0 flex-1 flex-col">
			{children ?? null}
		</main>
	);
};

Page.Section = ({ children }: { children?: React.ReactNode }) => {
	return <section>{children ?? null}</section>;
};

Page.Aside = ({ children }: { children?: React.ReactNode }) => {
	return <aside className="shrink-0">{children ?? null}</aside>;
};

Page.Footer = ({ children: _children }: { children?: React.ReactNode }) => {
	return <footer className="shrink-0"></footer>;
};
