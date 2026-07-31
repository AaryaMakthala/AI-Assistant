"use client";

/**
 * Sign in and sign up.
 *
 * One form for both, toggled rather than split across two routes: the fields are identical
 * and the distinction is one call. Two routes would double the copy that has to stay in
 * step for no gain in clarity.
 *
 * Errors are shown as Supabase words them. Rewriting them tends to blur the two cases that
 * matter — wrong password versus unconfirmed email — into a single unhelpful "sign-in
 * failed", and the second one is only solvable if the user is told which it is.
 */

import { Loader2, LogIn } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { getSupabaseClient, isSupabaseConfigured } from "@/lib/supabase/client";
import { cn } from "@/lib/utils";

type Mode = "signin" | "signup";

export default function LoginPage() {
  const router = useRouter();
  const supabase = getSupabaseClient();

  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [orgName, setOrgName] = useState("");
  const [error, setError] = useState<string | undefined>();
  const [notice, setNotice] = useState<string | undefined>();
  const [isBusy, setIsBusy] = useState(false);

  if (!isSupabaseConfigured() || !supabase) {
    return (
      <Shell>
        <p role="alert" className="text-sm text-danger">
          Sign-in is unavailable: this deployment has no Supabase credentials.
          Set <code className="font-mono">NEXT_PUBLIC_SUPABASE_URL</code> and{" "}
          <code className="font-mono">NEXT_PUBLIC_SUPABASE_ANON_KEY</code>.
        </p>
      </Shell>
    );
  }

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(undefined);
    setNotice(undefined);
    setIsBusy(true);

    try {
      if (mode === "signin") {
        const { error: failure } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (failure) {
          setError(failure.message);
          return;
        }
        // replace, not push: the login page should not be a back-button destination once
        // the session exists.
        router.replace("/");
        return;
      }

      const { data, error: failure } = await supabase.auth.signUp({
        email,
        password,
        // Read by the provisioning trigger (migration 0004) to name the new organization.
        // Display text only — it never selects an existing org.
        options: { data: { org_name: orgName.trim() || undefined } },
      });
      if (failure) {
        setError(failure.message);
        return;
      }
      // No session means the project requires email confirmation. Saying so is the whole
      // reason to distinguish the two: the account exists and the user must go read email.
      if (!data.session) {
        setNotice("Check your email to confirm your account, then sign in.");
        setMode("signin");
        return;
      }
      router.replace("/");
    } catch {
      setError("Could not reach the authentication service.");
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <Shell>
      <form onSubmit={submit} className="space-y-3">
        <Field
          label="Email"
          type="email"
          value={email}
          onChange={setEmail}
          autoComplete="email"
          required
        />
        <Field
          label="Password"
          type="password"
          value={password}
          onChange={setPassword}
          autoComplete={mode === "signin" ? "current-password" : "new-password"}
          required
        />
        {mode === "signup" && (
          <Field
            label="Organization name"
            type="text"
            value={orgName}
            onChange={setOrgName}
            autoComplete="organization"
            hint="Names your new workspace. Leave blank to use your email name."
          />
        )}

        {error && (
          <p role="alert" className="text-xs text-danger">
            {error}
          </p>
        )}
        {notice && (
          <p role="status" className="text-xs text-accent">
            {notice}
          </p>
        )}

        <button
          type="submit"
          disabled={isBusy}
          className={cn(
            "flex w-full items-center justify-center gap-2 rounded-md bg-accent px-3 py-2",
            "text-sm font-medium text-accent-foreground transition-colors",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            "disabled:opacity-60",
          )}
        >
          {isBusy ? (
            <Loader2 className="size-4 animate-spin" aria-hidden />
          ) : (
            <LogIn className="size-4" aria-hidden />
          )}
          {mode === "signin" ? "Sign in" : "Create account"}
        </button>
      </form>

      <button
        type="button"
        onClick={() => {
          setMode(mode === "signin" ? "signup" : "signin");
          setError(undefined);
          setNotice(undefined);
        }}
        className="mt-4 text-xs text-muted hover:text-foreground"
      >
        {mode === "signin"
          ? "No account? Create one"
          : "Already have an account? Sign in"}
      </button>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main className="flex min-h-dvh items-center justify-center bg-background px-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-surface p-6">
        <h1 className="text-base font-semibold">Knowledge Assistant</h1>
        <p className="mt-1 mb-5 text-xs text-muted">
          Sign in to reach your organization&apos;s documents and data.
        </p>
        {children}
      </div>
    </main>
  );
}

function Field({
  label,
  type,
  value,
  onChange,
  autoComplete,
  required,
  hint,
}: {
  label: string;
  type: string;
  value: string;
  onChange: (value: string) => void;
  autoComplete?: string;
  required?: boolean;
  hint?: string;
}) {
  return (
    <label className="block space-y-1">
      <span className="text-xs font-medium text-muted">{label}</span>
      <input
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        autoComplete={autoComplete}
        required={required}
        className={cn(
          "w-full rounded-md border border-border bg-surface-raised px-3 py-2 text-sm",
          "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
        )}
      />
      {hint && <span className="block text-[11px] text-muted">{hint}</span>}
    </label>
  );
}
