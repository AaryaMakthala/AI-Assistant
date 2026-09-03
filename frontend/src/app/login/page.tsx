"use client";

import {
  Building2,
  Loader2,
  LogIn,
  Eye,
  EyeOff,
  MessageSquareText,
  FileText,
  Sparkles,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import Link from "next/link";
import {
  checkEmail,
  createWorkspace,
  enterDemo,
  listWorkspaces,
} from "@/lib/api";
import { getSupabaseClient, isSupabaseConfigured } from "@/lib/supabase/client";
import { cn } from "@/lib/utils";

type Mode = "signin" | "signup";

/** Kept in step with the reset-password form so one flow cannot accept what the other rejects. */
const MIN_PASSWORD_LENGTH = 8;

/**
 * Where to land after signing in. `proxy.ts` puts the originally requested path in `next`,
 * and it is re-validated here as a same-origin absolute path: the redirect is driven by a
 * query parameter, so an unchecked value is an open redirect.
 */
function postSignInTarget(): string {
  const next = new URLSearchParams(window.location.search).get("next");
  if (!next || !next.startsWith("/") || next.startsWith("//")) return "/";
  return next;
}

export default function LoginPage() {
  const router = useRouter();
  const supabase = getSupabaseClient();

  const [mode, setMode] = useState<Mode>("signin");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [orgName, setOrgName] = useState("");

  const [error, setError] = useState<string | undefined>();
  const [isUnverified, setIsUnverified] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);
  const [isBusy, setIsBusy] = useState(false);

  // Zero-workspace state: user authenticated but has no organizations.
  const [needsOrg, setNeedsOrg] = useState(false);
  const [newOrgName, setNewOrgName] = useState("");
  const [orgSessionToken, setOrgSessionToken] = useState<string | undefined>();
  const [isCreatingOrg, setIsCreatingOrg] = useState(false);
  const [orgError, setOrgError] = useState<string | undefined>();

  // Password visibility
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Demo state
  const [isEnteringDemo, setIsEnteringDemo] = useState(false);
  const [demoError, setDemoError] = useState<string | undefined>();

  // Org creation state (after zero-workspace check)
  const [orgSuccess, setOrgSuccess] = useState(false);

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

  const handleResend = async () => {
    setIsBusy(true);
    setError(undefined);
    setResendSuccess(false);
    try {
      const { error: resendError } = await supabase.auth.resend({
        type: "signup",
        email,
        options: {
          emailRedirectTo: `${window.location.origin}/auth/callback`,
        },
      });
      if (resendError) {
        setError(resendError.message);
      } else {
        setResendSuccess(true);
      }
    } catch {
      setError(
        "Could not reach the authentication service. Please check your connection.",
      );
    } finally {
      setIsBusy(false);
    }
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(undefined);
    setIsUnverified(false);
    setResendSuccess(false);

    if (mode === "signup") {
      if (password !== confirmPassword) {
        setError("Passwords do not match.");
        return;
      }
      if (password.length < MIN_PASSWORD_LENGTH) {
        setError(
          `Password must be at least ${MIN_PASSWORD_LENGTH} characters long.`,
        );
        return;
      }
    }

    setIsBusy(true);

    try {
      if (mode === "signin") {
        // Check if the email is registered before attempting sign-in.
        // Supabase returns the same generic "Invalid login credentials" for both
        // wrong email and wrong password, so we need to distinguish them.
        let emailExists = true;
        try {
          const result = await checkEmail(email);
          emailExists = result.exists;
        } catch {
          // If the check fails (network, backend down), proceed with sign-in
          // anyway — the normal error handling will catch any issues.
          emailExists = true;
        }

        if (!emailExists) {
          setError("No account found with this email address.");
          return;
        }

        const { data: signInData, error: failure } =
          await supabase.auth.signInWithPassword({
            email,
            password,
          });

        if (failure) {
          const msg = failure.message;
          if (msg.toLowerCase().includes("email not confirmed")) {
            setIsUnverified(true);
            setError("Your email address has not been verified.");
          } else if (msg.includes("Invalid login credentials")) {
            setError("Incorrect password. Please try again.");
          } else {
            setError(msg);
          }
          return;
        }

        // Check whether the user has at least one organization/workspace.
        try {
          const token = signInData.session?.access_token;
          if (token) {
            const { workspaces } = await listWorkspaces({ token });
            if (workspaces.length === 0) {
              setOrgSessionToken(token);
              setNeedsOrg(true);
              return;
            }
          }
        } catch {
          // If the workspace check fails, proceed with the normal redirect.
        }

        router.replace(postSignInTarget());
        return;
      }

      const { data, error: failure } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: `${window.location.origin}/auth/callback`,
          data: {
            full_name: fullName.trim() || undefined,
            org_name: orgName.trim() || undefined,
          },
        },
      });

      if (failure) {
        const msg = failure.message;
        if (msg.includes("User already registered")) {
          setError(
            "An account with this email already exists. Try signing in instead.",
          );
        } else if (msg.includes("Password should be at least 6 characters")) {
          setError("Password must be at least 6 characters long.");
        } else {
          setError(msg);
        }
        return;
      }

      if (!data.session) {
        router.replace(
          `/check-email?email=${encodeURIComponent(email)}`,
        );
        return;
      }
      router.replace("/");
    } catch {
      setError(
        "Could not reach the authentication service. Please check your connection.",
      );
    } finally {
      setIsBusy(false);
    }
  };

  const toggleMode = () => {
    setMode(mode === "signin" ? "signup" : "signin");
    setError(undefined);
    setIsUnverified(false);
    setResendSuccess(false);
  };

  const handleDemo = async () => {
    setIsEnteringDemo(true);
    setDemoError(undefined);
    try {
      const demo = await enterDemo();
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: demo.email,
        password: demo.password,
      });
      if (signInError) {
        setDemoError("Could not start demo session. Please try again.");
        return;
      }
      router.replace(demo.redirect_url);
    } catch {
      setDemoError("Could not reach the demo server. Please try again.");
    } finally {
      setIsEnteringDemo(false);
    }
  };

  const handleCreateOrg = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!orgSessionToken || !newOrgName.trim()) return;
    setIsCreatingOrg(true);
    setOrgError(undefined);
    try {
      await createWorkspace(newOrgName.trim(), { token: orgSessionToken });
      setOrgSuccess(true);
    } catch (err) {
      setOrgError(
        err instanceof Error ? err.message : "Failed to create organization.",
      );
    } finally {
      setIsCreatingOrg(false);
    }
  };

  // Zero-workspace state: user is authenticated but has no organizations.
  if (needsOrg) {
    return (
      <Shell>
        <div className="space-y-4">
          {orgSuccess ? (
            <div className="rounded-md bg-accent/10 p-3 text-sm text-accent border border-accent/20">
              <p role="status">
                Organization created. We&apos;ve sent a verification email to
                your email address.
              </p>
              <button
                type="button"
                onClick={() => router.replace(postSignInTarget())}
                className="mt-2 text-xs font-medium underline hover:text-accent/80"
              >
                Continue to application
              </button>
            </div>
          ) : (
            <>
              <div className="rounded-md bg-warning/10 p-3 text-sm text-warning border border-warning/20">
                <p role="alert">
                  No organization is associated with this account.
                </p>
                <p className="mt-1 text-xs text-warning/80">
                  Please contact your administrator or create an organization to
                  continue.
                </p>
              </div>
              <form onSubmit={handleCreateOrg} className="space-y-4">
                <Field
                  label="Organization name"
                  type="text"
                  value={newOrgName}
                  onChange={setNewOrgName}
                  autoComplete="organization"
                  required
                />
                {orgError && (
                  <div className="rounded-md bg-danger/10 p-3 text-sm text-danger border border-danger/20">
                    <p role="alert">{orgError}</p>
                  </div>
                )}
                <button
                  type="submit"
                  disabled={isCreatingOrg || !newOrgName.trim()}
                  className={cn(
                    "flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2.5",
                    "text-sm font-semibold text-accent-foreground shadow-sm transition-all hover:bg-accent/90 active:scale-[0.98]",
                    "focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:outline-none",
                    "disabled:opacity-60 disabled:pointer-events-none",
                  )}
                >
                  {isCreatingOrg ? (
                    <Loader2 className="size-4 animate-spin" aria-hidden />
                  ) : (
                    <Building2 className="size-4" aria-hidden />
                  )}
                  Create organization
                </button>
              </form>
            </>
          )}
          <div className="text-center">
            <button
              type="button"
              onClick={() => {
                setNeedsOrg(false);
                setNewOrgName("");
                setOrgError(undefined);
                setOrgSuccess(false);
              }}
              className="text-sm font-medium text-muted hover:text-foreground transition-colors"
            >
              Back to sign in
            </button>
          </div>
        </div>
      </Shell>
    );
  }

  return (
    <div className="flex min-h-dvh">
      {/* Left panel — auth form */}
      <div className="flex flex-1 items-center justify-center bg-background px-4 py-12 sm:px-6 lg:px-8">
        <div className="w-full max-w-md animate-fade-in">
          {/* Logo + wordmark */}
          <div className="mb-8 flex items-center gap-3">
            <img
              src="/logo.png"
              alt="Office Brain logo"
              className="size-10 rounded-lg"
            />
            <span className="text-xl font-bold tracking-tight text-foreground">
              Office Brain
            </span>
          </div>

          {/* Auth card */}
          <div className="rounded-2xl border border-border bg-surface shadow-xl overflow-hidden">
            <div className="h-1 bg-gradient-to-r from-accent/40 via-accent to-accent/40" />
            <div className="px-8 pt-8 pb-8">
              <h1 className="text-2xl font-bold tracking-tight text-foreground">
                {mode === "signin" ? "Welcome back" : "Create your account"}
              </h1>
              <p className="mt-2 text-sm text-muted">
                {mode === "signin"
                  ? "Sign in to access your organization's knowledge."
                  : "Get started with Office Brain for your team."}
              </p>

              <form onSubmit={submit} className="mt-6 space-y-4">
                {mode === "signup" && (
                  <Field
                    label="Full name"
                    type="text"
                    value={fullName}
                    onChange={setFullName}
                    autoComplete="name"
                    required
                  />
                )}

                <Field
                  label="Email"
                  type="email"
                  value={email}
                  onChange={setEmail}
                  autoComplete="email"
                  required
                />

                <div className="space-y-1">
                  <PasswordField
                    label="Password"
                    value={password}
                    onChange={setPassword}
                    autoComplete={
                      mode === "signin" ? "current-password" : "new-password"
                    }
                    required
                    showPassword={showPassword}
                    setShowPassword={setShowPassword}
                  />
                  {mode === "signin" && (
                    <div className="flex justify-end pt-1">
                      <Link
                        href="/forgot-password"
                        className="text-xs text-muted hover:text-foreground hover:underline transition-colors"
                      >
                        Forgot password?
                      </Link>
                    </div>
                  )}
                  {mode === "signup" && password.length > 0 && (
                    <PasswordStrengthIndicator password={password} />
                  )}
                </div>

                {mode === "signup" && (
                  <>
                    <PasswordField
                      label="Confirm Password"
                      value={confirmPassword}
                      onChange={setConfirmPassword}
                      autoComplete="new-password"
                      required
                      showPassword={showConfirmPassword}
                      setShowPassword={setShowConfirmPassword}
                    />
                    <Field
                      label="Organization name"
                      type="text"
                      value={orgName}
                      onChange={setOrgName}
                      autoComplete="organization"
                      hint="Names your new workspace. Leave blank to use your name."
                    />
                  </>
                )}

                {error && (
                  <div className="rounded-md bg-danger/10 p-3 text-sm text-danger border border-danger/20">
                    <p role="alert">{error}</p>
                    {isUnverified && (
                      <button
                        type="button"
                        onClick={handleResend}
                        disabled={isBusy}
                        className="mt-2 text-xs font-medium underline hover:text-danger/80"
                      >
                        Resend verification email
                      </button>
                    )}
                  </div>
                )}

                {resendSuccess && (
                  <p
                    role="status"
                    className="rounded-md bg-accent/10 p-3 text-sm text-accent border border-accent/20"
                  >
                    Verification email sent. Please check your inbox.
                  </p>
                )}

                <button
                  type="submit"
                  disabled={isBusy}
                  className={cn(
                    "flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2.5",
                    "text-sm font-semibold text-accent-foreground shadow-sm transition-all hover:bg-accent/90 active:scale-[0.98]",
                    "focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:outline-none",
                    "disabled:opacity-60 disabled:pointer-events-none",
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

              <div className="mt-6 text-center">
                <button
                  type="button"
                  onClick={toggleMode}
                  className="text-sm font-medium text-muted hover:text-foreground transition-colors"
                >
                  {mode === "signin"
                    ? "Don't have an account? Sign up"
                    : "Already have an account? Sign in"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right panel — demo CTA */}
      <div className="hidden lg:flex flex-1 items-center justify-center bg-accent-subtle/30 px-12">
        <div className="w-full max-w-lg animate-fade-in">
          <div className="space-y-8">
            {/* Headline */}
            <div>
              <h2 className="text-3xl font-bold tracking-tight text-foreground">
                See Office Brain in action
              </h2>
              <p className="mt-3 text-base text-muted leading-relaxed">
                Ask questions about company documents, get instant answers with
                citations, and explore how AI-powered knowledge retrieval works
                for your team.
              </p>
            </div>

            {/* Feature preview — decorative chat bubbles */}
            <div className="space-y-3">
              <ChatBubble
                role="user"
                text="What's our annual leave policy?"
              />
              <ChatBubble
                role="assistant"
                text="Full-time employees accrue 15 days per year, increasing to 20 days after 3 years. Up to 5 unused days can carry over to Q1."
                sources={["Leave Policy · Page 1"]}
              />
            </div>

            {/* Feature highlights */}
            <div className="space-y-3">
              <FeatureItem
                icon={<MessageSquareText className="size-4" />}
                text="Natural language search across all company docs"
              />
              <FeatureItem
                icon={<FileText className="size-4" />}
                text="Instant citations so you always know the source"
              />
              <FeatureItem
                icon={<Sparkles className="size-4" />}
                text="No account required — jump right in"
              />
            </div>

            {/* Demo button */}
            <div className="space-y-3">
              <button
                type="button"
                onClick={handleDemo}
                disabled={isEnteringDemo}
                className={cn(
                  "flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-6 py-3",
                  "text-sm font-semibold text-accent-foreground shadow-md transition-all hover:bg-accent/90 active:scale-[0.98]",
                  "focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:outline-none",
                  "disabled:opacity-60 disabled:pointer-events-none",
                )}
              >
                {isEnteringDemo ? (
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                ) : (
                  <Sparkles className="size-4" aria-hidden />
                )}
                {isEnteringDemo ? "Starting demo..." : "Try the demo"}
              </button>
              {demoError && (
                <p className="text-xs text-center text-danger">{demoError}</p>
              )}
              <p className="text-xs text-center text-muted">
                A pre-loaded demo workspace with sample company documents.
                No account needed.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile demo CTA — shown below login form on narrow viewports */}
      <div className="lg:hidden px-4 pb-8 bg-accent-subtle/30">
        <div className="mx-auto max-w-md space-y-4 pt-4">
          <div className="h-px bg-border" />
          <div className="text-center space-y-2">
            <h2 className="text-lg font-semibold text-foreground">
              Try Office Brain
            </h2>
            <p className="text-sm text-muted">
              Explore a demo workspace with sample company documents.
            </p>
          </div>
          <button
            type="button"
            onClick={handleDemo}
            disabled={isEnteringDemo}
            className={cn(
              "flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2.5",
              "text-sm font-semibold text-accent-foreground shadow-sm transition-all hover:bg-accent/90 active:scale-[0.98]",
              "focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:outline-none",
              "disabled:opacity-60 disabled:pointer-events-none",
            )}
          >
            {isEnteringDemo ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : (
              <Sparkles className="size-4" aria-hidden />
            )}
            {isEnteringDemo ? "Starting demo..." : "Try the demo"}
          </button>
          {demoError && (
            <p className="text-xs text-center text-danger">{demoError}</p>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shell — wraps the zero-workspace and unconfigured states
// ---------------------------------------------------------------------------

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main className="flex min-h-dvh items-center justify-center bg-background/50 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-surface via-background to-background px-4 py-12 sm:px-6 lg:px-8">
      <div className="w-full max-w-md animate-fade-in relative overflow-hidden rounded-2xl bg-surface shadow-xl border border-border/50">
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-accent/40 via-accent to-accent/40 animate-pulse" />

        <div className="px-8 pt-10 pb-8">
          <div className="mb-8 flex items-center justify-center gap-3">
            <img
              src="/logo.png"
              alt="Office Brain logo"
              className="size-10 rounded-lg"
            />
            <span className="text-xl font-bold tracking-tight text-foreground">
              Office Brain
            </span>
          </div>
          {children}
        </div>
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Decorative chat bubble for the demo preview
// ---------------------------------------------------------------------------

function ChatBubble({
  role,
  text,
  sources,
}: {
  role: "user" | "assistant";
  text: string;
  sources?: string[];
}) {
  return (
    <div
      className={cn(
        "rounded-xl px-4 py-3 text-sm leading-relaxed max-w-sm",
        role === "user"
          ? "bg-accent/10 text-foreground ml-auto border border-accent/20"
          : "bg-surface border border-border text-foreground",
      )}
    >
      <p>{text}</p>
      {sources && sources.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {sources.map((s) => (
            <span
              key={s}
              className="inline-flex items-center gap-1 rounded-full bg-accent/10 px-2 py-0.5 text-[11px] font-medium text-accent"
            >
              <FileText className="size-3" aria-hidden />
              {s}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function FeatureItem({
  icon,
  text,
}: {
  icon: React.ReactNode;
  text: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-md bg-accent/10 text-accent">
        {icon}
      </div>
      <p className="text-sm text-muted leading-snug">{text}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Form field components (shared with the auth form)
// ---------------------------------------------------------------------------

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
    <div className="space-y-1.5">
      <label className="block text-sm font-medium text-foreground">
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        autoComplete={autoComplete}
        required={required}
        className={cn(
          "block w-full rounded-lg border border-border bg-surface-raised px-3 py-2 text-sm text-foreground",
          "placeholder:text-muted/60 transition-colors",
          "focus:border-accent focus:ring-1 focus:ring-accent focus:outline-none",
        )}
      />
      {hint && (
        <p className="text-[11px] text-muted pt-0.5">{hint}</p>
      )}
    </div>
  );
}

function PasswordField({
  label,
  value,
  onChange,
  autoComplete,
  required,
  showPassword,
  setShowPassword,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  autoComplete?: string;
  required?: boolean;
  showPassword: boolean;
  setShowPassword: (show: boolean) => void;
}) {
  return (
    <div className="space-y-1.5">
      <label className="block text-sm font-medium text-foreground">
        {label}
      </label>
      <div className="relative">
        <input
          type={showPassword ? "text" : "password"}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          autoComplete={autoComplete}
          required={required}
          className={cn(
            "block w-full rounded-lg border border-border bg-surface-raised pl-3 pr-10 py-2 text-sm text-foreground",
            "placeholder:text-muted/60 transition-colors",
            "focus:border-accent focus:ring-1 focus:ring-accent focus:outline-none",
          )}
        />
        <button
          type="button"
          className="absolute inset-y-0 right-0 flex items-center pr-3 text-muted hover:text-foreground transition-colors focus:outline-none"
          onClick={() => setShowPassword(!showPassword)}
          tabIndex={-1}
        >
          {showPassword ? (
            <EyeOff className="h-4 w-4" aria-hidden="true" />
          ) : (
            <Eye className="h-4 w-4" aria-hidden="true" />
          )}
        </button>
      </div>
    </div>
  );
}

function PasswordStrengthIndicator({ password }: { password: string }) {
  let score = 0;

  if (password.length > 5) score += 1;
  if (password.length > 7) score += 1;

  let variety = 0;
  if (/[a-z]/.test(password)) variety += 1;
  if (/[A-Z]/.test(password)) variety += 1;
  if (/[0-9]/.test(password)) variety += 1;
  if (/[^a-zA-Z0-9]/.test(password)) variety += 1;

  if (variety >= 2) score += 1;
  if (variety >= 4) score += 1;

  let strength = "Weak";
  let colorClass = "bg-danger";
  let width = "25%";

  if (score >= 4) {
    strength = "Very strong";
    colorClass = "bg-green-500";
    width = "100%";
  } else if (score === 3) {
    strength = "Strong";
    colorClass = "bg-yellow-500";
    width = "75%";
  } else if (score === 2) {
    strength = "Fair";
    colorClass = "bg-orange-500";
    width = "50%";
  }

  return (
    <div className="pt-1 space-y-1.5">
      <div className="h-1.5 w-full bg-border rounded-full overflow-hidden">
        <div
          className={cn(
            "h-full transition-all duration-300 ease-in-out",
            colorClass,
          )}
          style={{ width }}
        />
      </div>
      <p className="text-[11px] font-medium text-muted flex justify-between">
        <span>Password strength:</span>
        <span
          className={cn(
            strength === "Weak" && "text-danger",
            strength === "Fair" && "text-orange-500",
            strength === "Strong" && "text-yellow-500",
            strength === "Very strong" && "text-green-500",
          )}
        >
          {strength}
        </span>
      </p>
    </div>
  );
}
