import { Contrast, Monitor, Moon, Sun, SunDim } from "lucide-react";
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
export function ModeToggle() {
	const { theme, setTheme, contrast, setContrast } = useTheme();

	return (
		<Menu>
			<MenuTrigger
				render={
					<Button
						aria-label="Toggle theme"
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
					Light
				</MenuItem>
				<MenuItem onClick={() => setTheme("dim")}>
					<SunDim />
					Dim
				</MenuItem>
				<MenuItem onClick={() => setTheme("dark")}>
					<Moon />
					Dark
				</MenuItem>
				<MenuItem onClick={() => setTheme("system")}>
					<Monitor />
					System
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
						High contrast
					</span>
				</MenuCheckboxItem>
			</MenuPopup>
		</Menu>
	);
}
