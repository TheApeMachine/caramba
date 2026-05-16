import {
	createContext,
	useCallback,
	useContext,
	useEffect,
	useMemo,
	useState,
} from "react";

export type Theme = "light" | "dim" | "dark" | "system";

type ThemeContextValue = {
	theme: Theme;
	resolvedTheme: Exclude<Theme, "system">;
	setTheme: (theme: Theme) => void;
	contrast: boolean;
	setContrast: (contrast: boolean) => void;
};

const STORAGE_KEY_THEME = "caramba.theme";
const STORAGE_KEY_CONTRAST = "caramba.contrast";
const THEME_CLASSES: Exclude<Theme, "system">[] = ["light", "dim", "dark"];

const ThemeContext = createContext<ThemeContextValue | null>(null);

function readStoredTheme(): Theme {
	if (typeof window === "undefined") return "dark";
	const stored = window.localStorage.getItem(STORAGE_KEY_THEME);
	if (stored === "light" || stored === "dim" || stored === "dark" || stored === "system") {
		return stored;
	}
	return "dark";
}

function readStoredContrast(): boolean {
	if (typeof window === "undefined") return false;
	return window.localStorage.getItem(STORAGE_KEY_CONTRAST) === "1";
}

function resolveSystemTheme(): Exclude<Theme, "system"> {
	if (typeof window === "undefined") return "dark";
	return window.matchMedia("(prefers-color-scheme: dark)").matches
		? "dark"
		: "light";
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
	const [theme, setThemeState] = useState<Theme>(() => readStoredTheme());
	const [contrast, setContrastState] = useState<boolean>(() =>
		readStoredContrast(),
	);
	const [systemTheme, setSystemTheme] = useState<Exclude<Theme, "system">>(
		() => resolveSystemTheme(),
	);

	useEffect(() => {
		const media = window.matchMedia("(prefers-color-scheme: dark)");
		const update = () => setSystemTheme(media.matches ? "dark" : "light");
		media.addEventListener("change", update);
		return () => media.removeEventListener("change", update);
	}, []);

	const resolvedTheme = theme === "system" ? systemTheme : theme;

	useEffect(() => {
		const root = document.documentElement;
		for (const cls of THEME_CLASSES) {
			root.classList.toggle(cls, cls === resolvedTheme);
		}
		root.classList.toggle("contrast", contrast);
	}, [resolvedTheme, contrast]);

	const setTheme = useCallback((next: Theme) => {
		window.localStorage.setItem(STORAGE_KEY_THEME, next);
		setThemeState(next);
	}, []);

	const setContrast = useCallback((next: boolean) => {
		window.localStorage.setItem(STORAGE_KEY_CONTRAST, next ? "1" : "0");
		setContrastState(next);
	}, []);

	const value = useMemo(
		() => ({ theme, resolvedTheme, setTheme, contrast, setContrast }),
		[theme, resolvedTheme, setTheme, contrast, setContrast],
	);

	return (
		<ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
	);
}

export function useTheme(): ThemeContextValue {
	const ctx = useContext(ThemeContext);
	if (ctx === null) {
		throw new Error("useTheme must be used inside a ThemeProvider");
	}
	return ctx;
}
