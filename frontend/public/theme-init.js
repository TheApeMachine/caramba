(function () {
	try {
		var mode = localStorage.getItem("caramba.theme") || "dark";
		var contrastEnabled = localStorage.getItem("caramba.contrast") === "1";
		var visualTheme = localStorage.getItem("caramba.visual-theme") || "default";
		var root = document.documentElement;

		var themeMarkers = {
			neumorphic: "theme-neumorphic",
			glassmorphic: "theme-glassmorphic",
			"neo-brutalism": "theme-neo-brutalism",
			claymorphism: "theme-claymorphism",
			blueprint: "theme-blueprint",
			aurora: "theme-aurora",
		};

		var themeStylesheets = {
			neumorphic: "/themes/neumorphic.css",
			glassmorphic: "/themes/glassmorphic.css",
			"neo-brutalism": "/themes/neo-brutalism.css",
			claymorphism: "/themes/claymorphism.css",
			blueprint: "/themes/blueprint.css",
			aurora: "/themes/aurora.css",
		};

		["light", "dim", "dark"].forEach(function (modeClass) {
			root.classList.remove(modeClass);
		});

		Object.keys(themeMarkers).forEach(function (themeId) {
			root.classList.remove(themeMarkers[themeId]);
		});

		var resolvedMode =
			mode === "system"
				? matchMedia("(prefers-color-scheme: dark)").matches
					? "dark"
					: "light"
				: mode;

		root.classList.add(resolvedMode);
		root.classList.toggle("contrast", contrastEnabled);
		root.dataset.visualTheme = visualTheme;

		var markerClass = themeMarkers[visualTheme];
		if (markerClass) {
			root.classList.add(markerClass);
		}

		var stylesheetHref = themeStylesheets[visualTheme];
		if (stylesheetHref) {
			var link = document.createElement("link");
			link.id = "caramba-visual-theme";
			link.rel = "stylesheet";
			link.href = stylesheetHref;
			document.head.appendChild(link);
		}
	} catch (_error) {
		// Appearance bootstrap is best-effort before hydration.
	}
})();
