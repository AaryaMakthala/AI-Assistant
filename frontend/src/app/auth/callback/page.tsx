"use client";

/**
 * Where Supabase's emailed links land — verification, and the recovery link that leads to
 * a password reset.
 *
 * This page deliberately does *not* call `exchangeCodeForSession`. The browser client is
 * built by `@supabase/ssr` with `detectSessionInUrl` enabled, so it already consumes the
 * `?code=` during its own initialization. A second exchange would race the first and lose:
 * the PKCE verifier is single-use, so whichever call arrives second fails, and the user is
 * shown an authentication error for a link that actually worked.
 *
 * So the job here is only to wait for that built-in exchange to settle. `getSession()`
 * awaits the client's initialization promise, which makes it the join point.
 */

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2, AlertCircle } from "lucide-react";
import { getSupabaseClient } from "@/lib/supabase/client";

/** Only same-origin absolute paths, so a crafted link cannot bounce a fresh session offsite. */
function safeNext(value: string | null): string | null {
  if (!value || !value.startsWith("/") || value.startsWith("//")) return null;
  return value;
}

function AuthCallback() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    const run = async () => {
      // Supabase reports an unusable link (expired, already consumed) in the query string
      // rather than as a failed exchange, so this is checked before anything else.
      const described = searchParams.get("error_description") ?? searchParams.get("error");
      if (described) {
        if (active) setError(described);
        return;
      }

      const supabase = getSupabaseClient();
      if (!supabase) {
        if (active) {
          setError("This deployment has no Supabase credentials, so the link cannot be verified.");
        }
        return;
      }

      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!active) return;

      if (!session) {
        setError("This link is invalid or has expired. Request a new one and try again.");
        return;
      }

      // A recovery link produces a real session, which is what authorizes the password
      // change on the next page.
      const next = safeNext(searchParams.get("next"));
      router.replace(next ?? "/");
    };

    void run();
    return () => {
      active = false;
    };
  }, [router, searchParams]);

  if (error) {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-background p-4">
        <div className="w-full max-w-md rounded-2xl border border-border bg-surface-raised p-8 text-center shadow-xl">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-danger/10">
            <AlertCircle className="h-6 w-6 text-danger" aria-hidden />
          </div>
          <h1 className="mb-2 text-xl font-semibold text-foreground">Link could not be verified</h1>
          <p className="mb-6 text-sm text-muted">{error}</p>
          <button
            type="button"
            onClick={() => router.replace("/login")}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-accent-foreground transition-colors hover:bg-accent/90 focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none"
          >
            Back to sign in
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-dvh flex-col items-center justify-center gap-4 bg-background">
      <Loader2 className="size-10 animate-spin text-accent" aria-hidden />
      <p className="text-sm text-muted">Verifying your link…</p>
    </div>
  );
}

export default function AuthCallbackPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-dvh items-center justify-center bg-background">
          <Loader2 className="size-10 animate-spin text-accent" aria-hidden />
        </div>
      }
    >
      <AuthCallback />
    </Suspense>
  );
}
