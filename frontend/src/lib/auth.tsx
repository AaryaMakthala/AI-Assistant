"use client";

/**
 * Where the UI gets its bearer token and its role.
 *
 * The token is the live Supabase session's access token, refreshed by the client library
 * and re-published here through `onAuthStateChange`. Nothing caches it: an access token is
 * short-lived, and a copy held in a component would keep being sent after it expired.
 *
 * **Role is fetched from the backend, never decoded from the token.** A browser can read a
 * JWT's payload without verifying its signature, so a role obtained that way is a claim the
 * client is making about itself. It is indistinguishable, once assigned to a state
 * variable, from one the server actually vouched for. `/me` runs the same verification path
 * that guards every other endpoint, so the answer means something. What the UI does with it
 * is still only presentation — hiding a nav item is a courtesy, and the server rejects the
 * request regardless (CLAUDE.md 4.6).
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { getMe, type MeResponse } from "@/lib/api";
import { getSupabaseClient, isSupabaseConfigured } from "@/lib/supabase/client";

interface AuthValue {
  token?: string;
  isAuthenticated: boolean;
  /** False until the initial session lookup settles, so the UI can avoid deciding early. */
  isLoading: boolean;
  /** Server-confirmed identity. Undefined until `/me` answers. */
  me?: MeResponse;
  /** Server-confirmed role, or undefined while unknown. */
  role?: string | null;
  /** True when the deployment has no Supabase credentials — sign-in is impossible. */
  isConfigured: boolean;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthValue>({
  isAuthenticated: false,
  isLoading: true,
  isConfigured: false,
  signOut: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const supabase = getSupabaseClient();
  const configured = isSupabaseConfigured();

  const [token, setToken] = useState<string | undefined>();
  // Kept together so the identity is always attributable to the token it came from.
  const [fetchedMe, setFetchedMe] = useState<
    { token: string; me?: MeResponse } | undefined
  >();
  // Tracks whether the Supabase session has been resolved.  Separate from isLoading
  // because isLoading also covers the getMe phase (below).
  const [sessionChecked, setSessionChecked] = useState(false);

  useEffect(() => {
    if (!supabase) {
      setSessionChecked(true);
      return;
    }
    let active = true;

    // getSession() first: onAuthStateChange does not replay the session that already
    // exists, so a page load with a valid cookie would otherwise sit unauthenticated.
    void supabase.auth.getSession().then(({ data }) => {
      if (!active) return;
      setToken(data.session?.access_token);
      setSessionChecked(true);
    });

    const { data: subscription } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!active) return;
      setToken(session?.access_token);
      setSessionChecked(true);
    });

    return () => {
      active = false;
      subscription.subscription.unsubscribe();
    };
  }, [supabase]);

  useEffect(() => {
    if (!token) return;
    let active = true;
    // A failure here leaves `me` as undefined, which means the user is not
    // authenticated from the backend's perspective — even though they hold a
    // valid Supabase session.  The downstream `isAuthenticated` flag reflects
    // this, so protected pages redirect to /login rather than hammering the
    // backend with requests that will all be 403.
    void getMe({ token })
      .then((value) => {
        if (active) setFetchedMe({ token, me: value });
      })
      .catch(() => {
        if (active) setFetchedMe({ token, me: undefined });
      });
    return () => {
      active = false;
    };
  }, [token]);

  // Derived rather than cleared in an effect: pairing the result with the token it was
  // fetched for means a stale identity can never outlive the session it belonged to, and
  // there is no render in between where the old role is still visible.
  const me = fetchedMe && fetchedMe.token === token ? fetchedMe.me : undefined;

  const signOut = useCallback(async () => {
    await supabase?.auth.signOut();
    // Clearing the token is enough for this provider: `me` is derived from it, so the
    // identity goes with it.
    setToken(undefined);
    // A full navigation rather than a router push: it re-runs `proxy.ts` against the now
    // cleared cookies and drops every hook's in-memory state, so nothing from the previous
    // session survives into the login page.
    window.location.assign("/login");
  }, [supabase]);

  // isLoading: true until the full auth state is known.
  // 1. Supabase session has not been checked yet → loading.
  // 2. Token exists but getMe has not resolved for it yet → loading.
  //    (getMe resolves quickly; this prevents a flash-redirect to /login
  //    between session resolution and getMe resolution.)
  const isLoading =
    configured &&
    (!sessionChecked ||
      (token !== undefined && (!fetchedMe || fetchedMe.token !== token)));

  const value = useMemo<AuthValue>(
    () => ({
      token,
      // Authenticated only when BOTH the Supabase session exists AND the backend
      // confirmed the identity via /me.  If /me fails (403), the user has a
      // Supabase session but no valid workspace — they are redirected to /login.
      isAuthenticated: Boolean(token) && me !== undefined,
      isLoading,
      me,
      role: me?.role,
      isConfigured: configured,
      signOut,
    }),
    [token, isLoading, me, configured, signOut],
  );

  return <AuthContext value={value}>{children}</AuthContext>;
}

export function useAuth(): AuthValue {
  return useContext(AuthContext);
}

/** Just the token, for the common case of passing it to an API call. */
export function useAuthToken(): string | undefined {
  return useContext(AuthContext).token;
}
