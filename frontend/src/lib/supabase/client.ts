"use client";

/**
 * The browser's Supabase client.
 *
 * `createBrowserClient` from `@supabase/ssr` stores the session in cookies rather than
 * localStorage, which is what lets `proxy.ts` see whether a visitor is signed in before a
 * page renders. A localStorage session is invisible to the server, so route protection
 * would have to happen after hydration — a visible flash of the app for someone who is not
 * signed in.
 *
 * One instance per browser, memoized: each client opens its own auth listener and token
 * refresh timer, so constructing one per render would multiply both.
 */

import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

let client: SupabaseClient | undefined;

/** True when the app has been given somewhere to authenticate against. */
export function isSupabaseConfigured(): boolean {
  return Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);
}

/**
 * The shared client, or null when Supabase is not configured.
 *
 * Null rather than a throw: a missing key is a deployment mistake, and the honest response
 * is an app that renders and says it cannot sign anyone in — not a blank page from an
 * exception thrown during render.
 */
export function getSupabaseClient(): SupabaseClient | null {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) return null;
  client ??= createBrowserClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  return client;
}
