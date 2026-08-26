import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";

/**
 * Route protection.
 *
 * Next 16 renamed this file convention from `middleware` to `proxy`; the behaviour is the
 * same. It runs before a page renders, which is the point — a client-side redirect would
 * mean shipping and briefly displaying the workspace to someone with no session.
 *
 * This is an *optimistic* check, and deliberately shallow: it asks whether a session cookie
 * exists and resolves to a user, nothing more. It is not the authorization boundary. Every
 * request the app makes still carries a bearer token that the backend verifies against
 * Supabase's JWKS, and Postgres RLS constrains the rows underneath that (CLAUDE.md 4.6).
 * Someone who forges a cookie past this check reaches a UI whose every call returns 401.
 *
 * The cookie plumbing looks redundant but is not: `getUser()` may refresh an expired token,
 * and the refreshed cookies have to be written onto the response that is actually returned
 * or the new session is discarded the moment this function ends.
 */

/** Reachable without a session. */
const PUBLIC_PATHS = [
  "/login",
  "/check-email",
  "/forgot-password",
  "/reset-password",
  "/auth/callback",
];

/**
 * The subset of the above that a signed-in visitor has no reason to see, and is bounced
 * away from.
 *
 * `/reset-password` and `/auth/callback` are deliberately absent. Following a recovery
 * link *establishes a session* — that session is what authorizes the password change — so
 * treating "has a session" as "does not belong here" would redirect the user to the
 * workspace at exactly the moment they arrived to set a new password, making the reset
 * form unreachable for the only people who ever legitimately reach it.
 */
const SIGNED_IN_REDIRECT_PATHS = ["/login", "/check-email", "/forgot-password"];

const matches = (paths: string[], pathname: string) =>
  paths.some((path) => pathname === path || pathname.startsWith(`${path}/`));

export async function proxy(request: NextRequest) {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  // Unconfigured deployments are left alone rather than locked out: the app renders and
  // explains itself, which beats an infinite redirect to a login page that cannot work.
  if (!url || !anonKey) return NextResponse.next();

  let response = NextResponse.next({ request });

  const supabase = createServerClient(url, anonKey, {
    cookies: {
      getAll: () => request.cookies.getAll(),
      setAll: (cookies) => {
        for (const { name, value } of cookies) {
          request.cookies.set(name, value);
        }
        response = NextResponse.next({ request });
        for (const { name, value, options } of cookies) {
          response.cookies.set(name, value, options);
        }
      },
    },
  });

  // getUser(), not getSession(): getSession trusts the cookie's contents, while getUser
  // validates it with the auth server. Even for an optimistic check, the weaker one would
  // be satisfied by a cookie the visitor wrote themselves.
  const {
    data: { user },
  } = await supabase.auth.getUser();

  const { pathname } = request.nextUrl;
  const isPublic = matches(PUBLIC_PATHS, pathname);

  if (!user && !isPublic) {
    const target = request.nextUrl.clone();
    target.pathname = "/login";
    // Where to return to after signing in. Only ever a path from this same URL, so it
    // cannot be pointed at another origin.
    target.searchParams.set("next", pathname);
    return NextResponse.redirect(target);
  }

  if (user && matches(SIGNED_IN_REDIRECT_PATHS, pathname)) {
    const target = request.nextUrl.clone();
    target.pathname = "/";
    target.search = "";
    return NextResponse.redirect(target);
  }

  return response;
}

export const config = {
  // Static assets and image optimizer output never need a session check, and excluding
  // them keeps this off the hot path for every icon and font request.
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)"],
};
