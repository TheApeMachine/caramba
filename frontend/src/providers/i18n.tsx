import { useEffect } from "react";
import { useTranslation } from "react-i18next";

/*
I18nProvider syncs the document lang attribute when the active locale changes.
i18next is initialized at module load via #/i18n; this wrapper handles DOM updates.
*/
export const I18nProvider = ({ children }: { children: React.ReactNode }) => {
	const { i18n } = useTranslation();

	useEffect(() => {
		const syncLanguage = (language: string) => {
			const resolved = language.split("-")[0] ?? "en";
			document.documentElement.lang = resolved;
		};

		syncLanguage(i18n.resolvedLanguage ?? i18n.language);
		i18n.on("languageChanged", syncLanguage);

		return () => {
			i18n.off("languageChanged", syncLanguage);
		};
	}, [i18n]);

	return children;
};
