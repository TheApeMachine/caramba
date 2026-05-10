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
import { Drawer } from "#/components/ui/drawer";
import { Field } from "#/components/ui/field";
import { Input } from "#/components/ui/input";
import { Menu, MenuItem, MenuPopup, MenuTrigger } from "#/components/ui/menu";

export const Page = ({ children }: { children?: React.ReactNode }) => {
	return <>{children ?? null}</>;
};

Page.Header = ({ children }: { children?: React.ReactNode }) => {
	return (
		<header className="grid-area-header shrink-0 p-4">
			<Breadcrumb>
				<BreadcrumbList>
					<BreadcrumbItem>
						<Drawer position="left">
							<Drawer.Trigger render={<Button variant="outline" />}>
								<MenuIcon />
							</Drawer.Trigger>
							<Drawer.Popup variant="inset">
								<Drawer.Header>
									<Drawer.Title>Manage team member</Drawer.Title>
									<Drawer.Description>
										View and manage a user in your team.
									</Drawer.Description>
								</Drawer.Header>
								<Drawer.Panel className="grid gap-4">
									<div className="grid gap-1">
										<p className="text-muted-foreground text-sm">Name</p>
										<p className="font-medium text-sm">Bora Baloglu</p>
									</div>
									<div className="grid gap-1">
										<p className="text-muted-foreground text-sm">Email</p>
										<p className="font-medium text-sm">bora@example.com</p>
									</div>
								</Drawer.Panel>
								<Drawer.Footer>
									<Drawer position="right">
										<Drawer.Trigger render={<Button variant="outline" />}>
											Edit details
										</Drawer.Trigger>
										<Drawer.Popup variant="inset">
											<Drawer.Header>
												<Drawer.Title>Edit details</Drawer.Title>
												<Drawer.Description>
													Make changes to the member&apos;s information.
												</Drawer.Description>
											</Drawer.Header>
											<Drawer.Panel className="grid gap-4">
												<Field>
													<Field.Label>Name</Field.Label>
													<Input defaultValue="Bora Baloglu" type="text" />
												</Field>
												<Field>
													<Field.Label>Email</Field.Label>
													<Input defaultValue="bora@example.com" type="email" />
												</Field>
											</Drawer.Panel>
											<Drawer.Footer>
												<Drawer.Close render={<Button variant="ghost" />}>
													Cancel
												</Drawer.Close>
												<Button type="submit">Save changes</Button>
											</Drawer.Footer>
										</Drawer.Popup>
									</Drawer>
								</Drawer.Footer>
							</Drawer.Popup>
						</Drawer>
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
	return (
		<footer className="shrink-0">
			<Drawer>
				<Drawer.Trigger render={<Button variant="outline" />}>
					Nested drawers
				</Drawer.Trigger>
				<Drawer.Popup showBar>
					<Drawer.Header className="text-center">
						<Drawer.Title>First step</Drawer.Title>
						<Drawer.Description>
							This is the first step. Tap the button below to continue to the
							next screen.
						</Drawer.Description>
					</Drawer.Header>
					<Drawer.Footer
						className="justify-center sm:justify-center"
						variant="bare"
					>
						<Drawer.Close render={<Button variant="ghost" />}>
							Cancel
						</Drawer.Close>
						<Drawer>
							<Drawer.Trigger render={<Button variant="outline" />}>
								Continue
							</Drawer.Trigger>
							<Drawer.Popup showBar>
								<Drawer.Header className="text-center">
									<Drawer.Title>Second step</Drawer.Title>
									<Drawer.Description>
										You&apos;ve reached the second step. Tap the button below to
										continue to the next screen.
									</Drawer.Description>
								</Drawer.Header>
								<Drawer.Panel>
									<div className="flex justify-center">
										<div className="size-48 shrink-0 rounded-xl border bg-muted" />
									</div>
								</Drawer.Panel>
								<Drawer.Footer
									className="justify-center sm:justify-center"
									variant="bare"
								>
									<Drawer.Close render={<Button variant="ghost" />}>
										Back
									</Drawer.Close>
									<Drawer>
										<Drawer.Trigger render={<Button variant="outline" />}>
											Continue
										</Drawer.Trigger>
										<Drawer.Popup showBar>
											<Drawer.Header className="text-center">
												<Drawer.Title>Third step</Drawer.Title>
												<Drawer.Description>
													You&apos;ve reached the final step. You can close this
													drawer or go back.
												</Drawer.Description>
											</Drawer.Header>
											<Drawer.Panel>
												<div className="flex justify-center">
													<div className="size-32 shrink-0 rounded-full border bg-muted" />
												</div>
											</Drawer.Panel>
										</Drawer.Popup>
									</Drawer>
								</Drawer.Footer>
							</Drawer.Popup>
						</Drawer>
					</Drawer.Footer>
				</Drawer.Popup>
			</Drawer>
		</footer>
	);
};
