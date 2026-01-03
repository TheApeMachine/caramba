import { createContext, useContext, useEffect, useState } from "react";

type Theme = "dark" | "light" | "system";

type ThemeProviderProps = {
	children: React.ReactNode;
	defaultTheme?: Theme;
	storageKey?: string;
};

type ThemeProviderState = {
	theme: Theme;
	setTheme: (theme: Theme) => void;
};

const ThemeProviderContext = createContext<ThemeProviderState | undefined>(
	undefined,
);

export function ThemeProvider({
	children,
	defaultTheme = "system",
	storageKey = "vite-ui-theme",
	...props
}: ThemeProviderProps) {
	const [theme, setTheme] = useState<Theme>(() => {
		// Only access localStorage in browser
		if (typeof window !== "undefined") {
			return (localStorage.getItem(storageKey) as Theme) || defaultTheme;
		}
		return defaultTheme;
	});

	useEffect(() => {
		// Read persisted theme on mount (client-side only)
		if (typeof window !== "undefined") {
			const persisted = localStorage.getItem(storageKey) as Theme;
			if (persisted && persisted !== theme) {
				setTheme(persisted);
			}
		}
	}, [storageKey]);

	useEffect(() => {
		const root = window.document.documentElement;

		root.classList.remove("light", "dark");

		if (theme === "system") {
			const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
			const updateSystemTheme = () => {
				const systemTheme = mediaQuery.matches ? "dark" : "light";
				root.classList.add(systemTheme);
			};

			updateSystemTheme();
			mediaQuery.addEventListener("change", updateSystemTheme);

			return () => {
				mediaQuery.removeEventListener("change", updateSystemTheme);
			};
		}

		root.classList.add(theme);
	}, [theme]);

	const value = {
		theme,
		setTheme: (theme: Theme) => {
			localStorage.setItem(storageKey, theme);
			setTheme(theme);
		},
	};

	return (
		<ThemeProviderContext.Provider value={value} {...props}>
			{children}
		</ThemeProviderContext.Provider>
	);
}

export const useTheme = () => {
	const context = useContext(ThemeProviderContext);

	if (context === undefined)
		throw new Error("useTheme must be used within a ThemeProvider");

	return context;
};