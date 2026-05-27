(function () {
	try {
		var theme = localStorage.getItem("caramba.theme") || "dark";
		var contrastEnabled = localStorage.getItem("caramba.contrast") === "1";
		var root = document.documentElement;

		["light", "dim", "dark"].forEach(function (themeClass) {
			root.classList.remove(themeClass);
		});

		var resolvedTheme =
			theme === "system"
				? matchMedia("(prefers-color-scheme: dark)").matches
					? "dark"
					: "light"
				: theme;

		root.classList.add(resolvedTheme);
		root.classList.toggle("contrast", contrastEnabled);
	} catch (_error) {
		// Theme bootstrap is best-effort before hydration.
	}
})();
