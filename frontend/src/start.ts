import { clerkMiddleware } from "@clerk/tanstack-react-start/server";
import { createStart } from "@tanstack/react-start";

/*
startInstance registers global request middleware for TanStack Start (Clerk auth).
*/
export const startInstance = createStart(() => ({
	requestMiddleware: [
		clerkMiddleware({
			signInUrl: "/sign-in",
			signUpUrl: "/sign-up",
		}),
	],
}));
