"use client";

/**
 * Where the UI gets its bearer token.
 *
 * Phase 9 replaces the innards of this module with Supabase Auth. It exists now, ahead of
 * that, so every call site is already written against `useAuthToken()` — swapping the
 * implementation then is a change to one file rather than to every component that fetches.
 *
 * Until then the token comes from `NEXT_PUBLIC_DEV_JWT`, unset by default. That is a
 * deliberately awkward seam rather than a convenient one: it is build-time, visible in the
 * client bundle, and useless for anything but a locally minted development token. It must
 * never hold a real user's session — Phase 9 supplies those, per-session and out of the
 * bundle.
 *
 * With no token configured the app still renders. Requests come back 401 and surface as an
 * "authentication required" state, which is the honest thing to show for a backend that
 * requires a JWT (CLAUDE.md 4.6) and a frontend that cannot yet obtain one.
 */

import { createContext, useContext, useMemo, type ReactNode } from "react";

interface AuthValue {
  token?: string;
  /** False until Phase 9 wires up real sign-in; drives the "sign in required" state. */
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthValue>({ isAuthenticated: false });

export function AuthProvider({ children }: { children: ReactNode }) {
  const value = useMemo<AuthValue>(() => {
    const token = process.env.NEXT_PUBLIC_DEV_JWT || undefined;
    return { token, isAuthenticated: Boolean(token) };
  }, []);

  return <AuthContext value={value}>{children}</AuthContext>;
}

export function useAuth(): AuthValue {
  return useContext(AuthContext);
}

/** Just the token, for the common case of passing it to an API call. */
export function useAuthToken(): string | undefined {
  return useContext(AuthContext).token;
}
