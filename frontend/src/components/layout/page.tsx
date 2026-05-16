import { Link, useLocation } from "@tanstack/react-router";
import { Menu as MenuIcon } from "lucide-react";
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

const humanize = (segment: string) =>
	decodeURIComponent(segment)
		.replace(/[-_]+/g, " ")
		.replace(/\b\w/g, (character) => character.toUpperCase());

const useRouteCrumbs = () => {
	const { pathname } = useLocation();
	const segments = pathname.split("/").filter(Boolean);

	return segments.map((segment, index) => ({
		href: `/${segments.slice(0, index + 1).join("/")}`,
		label: humanize(segment),
	}));
};

const RouteCrumb = ({
	href,
	label,
	isLast,
}: {
	href: string;
	label: string;
	isLast: boolean;
}) => {
	return (
		<>
			<BreadcrumbSeparator />
			<BreadcrumbItem>
				{isLast ? (
					<BreadcrumbPage>{label}</BreadcrumbPage>
				) : (
					<BreadcrumbLink href={href}>{label}</BreadcrumbLink>
				)}
			</BreadcrumbItem>
		</>
	);
};

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
					{useRouteCrumbs().map((crumb, index, all) => (
						<RouteCrumb
							key={crumb.href}
							href={crumb.href}
							isLast={index === all.length - 1}
							label={crumb.label}
						/>
					))}
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
