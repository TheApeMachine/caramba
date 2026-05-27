import { Contrast, Layers, Monitor, Moon, Sun, SunDim } from "lucide-react";
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
import {
	type ColorMode,
	VISUAL_THEME_OPTIONS,
	type VisualTheme,
} from "#/lib/appearance";
import { useTheme } from "#/providers/theme";

const modeIcons: Record<ColorMode, React.ReactNode> = {
	light: <Sun className="size-4" />,
	dim: <SunDim className="size-4" />,
	dark: <Moon className="size-4" />,
	system: <Monitor className="size-4" />,
};

/*
ModeToggle exposes color mode (light/dim/dark/system), high contrast, and
visual theme styles. Modes set document classes; visual themes load optional
stylesheets.
*/
export const ModeToggle = () => {
	const { mode, setMode, contrast, setContrast, visualTheme, setVisualTheme } =
		useTheme();
	const { t } = useTranslation();

	const selectVisualTheme = (next: VisualTheme) => {
		if (next === visualTheme) return;
		setVisualTheme(next);
	};

	return (
		<Menu>
			<MenuTrigger
				render={
					<Button
						aria-label={t("mode.toggle")}
						size="icon"
						type="button"
						variant="outline"
					/>
				}
			>
				{modeIcons[mode]}
			</MenuTrigger>
			<MenuPopup align="end" className="min-w-52">
				<MenuItem onClick={() => setMode("light")}>
					<Sun />
					{t("mode.light")}
				</MenuItem>
				<MenuItem onClick={() => setMode("dim")}>
					<SunDim />
					{t("mode.dim")}
				</MenuItem>
				<MenuItem onClick={() => setMode("dark")}>
					<Moon />
					{t("mode.dark")}
				</MenuItem>
				<MenuItem onClick={() => setMode("system")}>
					<Monitor />
					{t("mode.system")}
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
						{t("mode.highContrast")}
					</span>
				</MenuCheckboxItem>
				<MenuSeparator />
				<div className="px-2 py-1.5 text-xs font-medium text-muted-foreground flex items-center gap-2">
					<Layers className="size-3.5 shrink-0" />
					{t("visualTheme.label")}
				</div>
				{VISUAL_THEME_OPTIONS.map((themeId) => (
					<MenuItem key={themeId} onClick={() => selectVisualTheme(themeId)}>
						{t(`visualTheme.${themeId}`)}
						{visualTheme === themeId ? " ✓" : ""}
					</MenuItem>
				))}
			</MenuPopup>
		</Menu>
	);
};
