import {
	Show,
	SignInButton,
	SignUpButton,
	UserButton,
} from "@clerk/tanstack-react-start";
import { Link } from "@tanstack/react-router";
import { LightbulbIcon } from "lucide-react";
import { OrganizationAccountMenu } from "#/components/auth/organization-account-menu";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import { useIsClerkAppAdmin } from "#/hooks/use-clerk-app-role";

/*
SessionControls surfaces sign-in and account UI in the shell header.
*/
export function SessionControls() {
	const isClerkAppAdmin = useIsClerkAppAdmin();

	return (
		<div className="ml-auto flex shrink-0 items-center gap-2">
			{isClerkAppAdmin ? <Badge variant="warning">Admin</Badge> : null}
			<Show when="signed-out">
				<SignInButton mode="modal">
					<Button size="sm" type="button" variant="outline">
						Sign in
					</Button>
				</SignInButton>
				<SignUpButton mode="modal">
					<Button size="sm" type="button" variant="default">
						Sign up
					</Button>
				</SignUpButton>
			</Show>
			<Show when="signed-in">
				<div className="flex shrink-0 items-center gap-2">
					<Link to="/request-feature">
						<Button size="sm" type="button" variant="outline">
							<LightbulbIcon aria-hidden className="size-4" />
							Request feature
						</Button>
					</Link>
					<OrganizationAccountMenu />
					<UserButton />
				</div>
			</Show>
		</div>
	);
}
