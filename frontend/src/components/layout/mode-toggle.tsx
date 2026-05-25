import { Contrast, Monitor, Moon, Sun, SunDim } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "#/components/ui/button";
import {
	Menu,
	MenuCheckboxItem,
	MenuItem,
	MenuPopup,
	MenuSeparator,
	MenuTrigger,
} from "#/components/ui/menu";
import { type Theme, useTheme } from "#/providers/theme";

const themeIcons: Record<Theme, React.ReactNode> = {
	light: <Sun className="size-4" />,
	dim: <SunDim className="size-4" />,
	dark: <Moon className="size-4" />,
	system: <Monitor className="size-4" />,
};

/*
ModeToggle exposes light/dim/dark/system theme selection and a high-contrast
toggle. State is owned by the ThemeProvider and persisted to localStorage.
*/
export const ModeToggle = () => {
	const { theme, setTheme, contrast, setContrast } = useTheme();
	const { t } = useTranslation();

	return (
		<Menu>
			<MenuTrigger
				render={
					<Button
						aria-label={t("theme.toggle")}
						size="icon"
						type="button"
						variant="outline"
					/>
				}
			>
				{themeIcons[theme]}
			</MenuTrigger>
			<MenuPopup align="end" className="min-w-48">
				<MenuItem onClick={() => setTheme("light")}>
					<Sun />
					{t("theme.light")}
				</MenuItem>
				<MenuItem onClick={() => setTheme("dim")}>
					<SunDim />
					{t("theme.dim")}
				</MenuItem>
				<MenuItem onClick={() => setTheme("dark")}>
					<Moon />
					{t("theme.dark")}
				</MenuItem>
				<MenuItem onClick={() => setTheme("system")}>
					<Monitor />
					{t("theme.system")}
				</MenuItem>
				<MenuSeparator />
				<MenuCheckboxItem
					checked={contrast}
					closeOnClick={false}
					onCheckedChange={setContrast}
					variant="switch"
				>
					<span className="flex items-center gap-2">
						<Contrast className="size-4" />
						{t("theme.highContrast")}
					</span>
				</MenuCheckboxItem>
			</MenuPopup>
		</Menu>
	);
};
