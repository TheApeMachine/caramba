import i18n from "i18next";
import LanguageDetector from "i18next-browser-languagedetector";
import { initReactI18next } from "react-i18next";
import de from "#/locales/de/translation";
import en from "#/locales/en/translation";

export const STORAGE_KEY_LANGUAGE = "caramba.language";

export const supportedLanguages = ["en", "de"] as const;
export type SupportedLanguage = (typeof supportedLanguages)[number];

const resources = {
	en: { translation: en },
	de: { translation: de },
} as const;

/*
initI18n configures react-i18next for the frontend shell.
Imported once from the root route so translations are ready before render.
*/
export function initI18n() {
	if (i18n.isInitialized) {
		return i18n;
	}

	i18n
		.use(LanguageDetector)
		.use(initReactI18next)
		.init({
			resources,
			fallbackLng: "en",
			lng: "en",
			supportedLngs: [...supportedLanguages],
			initImmediate: false,
			interpolation: {
				escapeValue: false,
			},
			detection: {
				order: ["localStorage", "navigator"],
				caches: ["localStorage"],
				lookupLocalStorage: STORAGE_KEY_LANGUAGE,
			},
		});

	return i18n;
}

export default initI18n();
