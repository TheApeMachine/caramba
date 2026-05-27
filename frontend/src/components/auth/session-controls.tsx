import {
	Show,
	SignInButton,
	SignUpButton,
	UserButton,
} from "@clerk/tanstack-react-start";
import { ClientOnly, Link } from "@tanstack/react-router";
import { LightbulbIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { WorkspaceSwitcher } from "#/components/auth/workspace-switcher";
import { LanguageToggle } from "#/components/layout/language-toggle";
import { ModeToggle } from "#/components/layout/mode-toggle";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import { useIsClerkAppAdmin } from "#/hooks/use-clerk-app-role";

/*
SessionControls surfaces sign-in and account UI in the shell header.
*/
export function SessionControls() {
	const isClerkAppAdmin = useIsClerkAppAdmin();
	const { t } = useTranslation();

	return (
		<div className="ml-auto flex shrink-0 items-center gap-2">
			{isClerkAppAdmin ? (
				<Badge variant="warning">{t("common.admin")}</Badge>
			) : null}
			<LanguageToggle />
			<ClientOnly fallback={null}>
				<ModeToggle />
			</ClientOnly>
			<Show when="signed-out">
				<SignInButton mode="modal">
					<Button size="sm" type="button" variant="outline">
						{t("auth.signIn")}
					</Button>
				</SignInButton>
				<SignUpButton mode="modal">
					<Button size="sm" type="button" variant="default">
						{t("auth.signUp")}
					</Button>
				</SignUpButton>
			</Show>
			<Show when="signed-in">
				<div className="flex shrink-0 items-center gap-2">
					<Link to="/request-feature">
						<Button size="sm" type="button" variant="outline">
							<LightbulbIcon aria-hidden className="size-4" />
							{t("auth.requestFeature")}
						</Button>
					</Link>
					<WorkspaceSwitcher />
					<UserButton />
				</div>
			</Show>
		</div>
	);
}
