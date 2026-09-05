"use client";

import {
  ArrowRight,
  Building2,
  Eye,
  EyeOff,
  Loader2,
  Play,
} from "lucide-react";
import Image from "next/image";
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
import "./login.css";

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
      <LoginShell
        title="Sign in to Office Brain"
        subtitle="Ask questions across your organization's approved knowledge — every answer grounded in your documents, with citations."
      >
        <p role="alert" className="login-alert login-alert-error">
          Sign-in is unavailable: this deployment has no Supabase credentials.
          Set <code className="font-mono">NEXT_PUBLIC_SUPABASE_URL</code> and{" "}
          <code className="font-mono">NEXT_PUBLIC_SUPABASE_ANON_KEY</code>.
        </p>
      </LoginShell>
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
      <LoginShell
        title="Create your organization"
        subtitle="You're signed in, but this account isn't part of a workspace yet."
      >
        <div className="space-y-5">
          {orgSuccess ? (
            <>
              <div className="login-alert login-alert-success">
                <p role="status">
                  Organization created. We&apos;ve sent a verification email to
                  your email address.
                </p>
              </div>
              <button
                type="button"
                onClick={() => router.replace(postSignInTarget())}
                className="login-btn login-btn-primary"
              >
                Continue to application
              </button>
            </>
          ) : (
            <>
              <div className="login-alert login-alert-warning">
                <p role="alert">
                  No organization is associated with this account.
                </p>
                <p className="mt-1 text-xs text-white/60">
                  Please contact your administrator or create an organization to
                  continue.
                </p>
              </div>
              <form onSubmit={handleCreateOrg} className="login-form">
                <div className="login-fields">
                  <Field
                    label="Organization name"
                    type="text"
                    value={newOrgName}
                    onChange={setNewOrgName}
                    autoComplete="organization"
                    required
                  />
                </div>
                {orgError && (
                  <div className="login-alert login-alert-error">
                    <p role="alert">{orgError}</p>
                  </div>
                )}
                <div className="login-actions">
                  <button
                    type="submit"
                    disabled={isCreatingOrg || !newOrgName.trim()}
                    className="login-btn login-btn-primary"
                  >
                    {isCreatingOrg ? (
                      <Loader2 className="size-4 animate-spin" aria-hidden />
                    ) : (
                      <Building2 className="size-4" aria-hidden />
                    )}
                    Create organization
                  </button>
                </div>
              </form>
              <div className="text-center">
                <button
                  type="button"
                  onClick={() => {
                    setNeedsOrg(false);
                    setNewOrgName("");
                    setOrgError(undefined);
                    setOrgSuccess(false);
                  }}
                  className="login-link font-medium"
                >
                  Back to sign in
                </button>
              </div>
            </>
          )}
        </div>
      </LoginShell>
    );
  }

  return (
    <LoginShell
      title={
        mode === "signin"
          ? "Sign in to Office Brain"
          : "Create your account"
      }
      subtitle={
        mode === "signin"
          ? "Ask questions across your organization's approved knowledge — every answer grounded in your documents, with citations."
          : "Start a workspace for your team and get grounded, cited answers from the documents that matter."
      }
      onDemo={handleDemo}
      demoBusy={isEnteringDemo}
    >
      <form onSubmit={submit} className="login-form">
        <div className="login-fields">
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
            <div className="flex justify-end">
              <Link
                href="/forgot-password"
                className="login-link hover:underline"
              >
                Forgot password?
              </Link>
            </div>
          )}

          {mode === "signup" && password.length > 0 && (
            <PasswordStrengthIndicator password={password} />
          )}

          {mode === "signup" && (
            <PasswordField
              label="Confirm password"
              value={confirmPassword}
              onChange={setConfirmPassword}
              autoComplete="new-password"
              required
              showPassword={showConfirmPassword}
              setShowPassword={setShowConfirmPassword}
            />
          )}

          {mode === "signup" && (
            <Field
              label="Organization name"
              type="text"
              value={orgName}
              onChange={setOrgName}
              autoComplete="organization"
              hint="Names your new workspace. Leave blank to use your name."
            />
          )}
        </div>

        {error && (
          <div className="login-alert login-alert-error">
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
            className="login-alert login-alert-success"
          >
            Verification email sent. Please check your inbox.
          </p>
        )}

        <div className="login-actions">
          <button
            type="submit"
            disabled={isBusy}
            className="login-btn login-btn-primary"
          >
            {isBusy ? (
              <>
                <Loader2 className="size-4 animate-spin" aria-hidden />
                Signing in...
              </>
            ) : (
              <>
                {mode === "signin" ? "Sign in" : "Create account"}
                <ArrowRight className="size-4" aria-hidden />
              </>
            )}
          </button>

          <button
            type="button"
            onClick={handleDemo}
            disabled={isEnteringDemo}
            className="login-btn login-btn-secondary"
          >
            {isEnteringDemo ? (
              <>
                <Loader2 className="size-4 animate-spin" aria-hidden />
                Starting demo...
              </>
            ) : (
              <>
                <Play className="size-4" aria-hidden />
                Try the demo
              </>
            )}
          </button>

          {demoError && (
            <p className="login-demo-error">{demoError}</p>
          )}

          <p className="text-center text-xs text-white/50">
            A pre-loaded demo workspace with sample company documents. No
            account needed.
          </p>

          <div className="pt-2 text-center">
            <button
              type="button"
              onClick={toggleMode}
              className="login-link font-medium"
            >
              {mode === "signin"
                ? "Don't have an account? Sign up"
                : "Already have an account? Sign in"}
            </button>
          </div>
        </div>
      </form>
    </LoginShell>
  );
}

// ---------------------------------------------------------------------------
// Shell — background layer, top nav, and content placement. Shared by the
// main sign-in/sign-up view, the zero-workspace state, and the unconfigured
// notice so they all sit on the same glass treatment.
// ---------------------------------------------------------------------------

function LoginShell({
  title,
  subtitle,
  onDemo,
  demoBusy,
  children,
}: {
  title?: string;
  subtitle?: string;
  onDemo?: () => void;
  demoBusy?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="login-page">
      {/* Full-bleed background: one next/image layer of the real photo
       * (public/login-bg.png), object-fit: cover — fills the entire viewport
       * edge-to-edge with zero letterboxing, seams, or masks. The photo's
       * own calm dark zone on the left is what the headline sits on; cover's
       * minor edge-crop (≈4% of width at desktop) never reaches it. */}
      <Image
        src="/login-bg.png"
        alt=""
        aria-hidden
        fill
        priority
        sizes="100vw"
        draggable={false}
        style={{ objectFit: "cover" }}
        className="login-bg"
      />

      {/* The three real blob renders, individually positioned and layered on
       * top of the background (z-index 1) — still behind the nav and glass
       * card (z-index 2), which blur whatever blob edge sits close enough to
       * the panel boundary. */}
      <div className="login-blobs" aria-hidden="true">
        {/* unoptimized so the DOM <img> srcs are the literal /blobs/*.png
         * paths (decorative PNGs, already lean — no re-encode needed). */}
        <Image
          src="/blobs/stone.png"
          alt=""
          width={677}
          height={369}
          unoptimized
          priority
          className="login-blob login-blob-gcircle"
        />
        <Image
          src="/blobs/green.png"
          alt=""
          width={500}
          height={500}
          unoptimized
          className="login-blob login-blob-green"
        />
        <Image
          src="/blobs/stone.png"
          alt=""
          width={677}
          height={369}
          unoptimized
          className="login-blob login-blob-stone"
        />
      </div>

      {/* Nav floats on the raw background, above the centered glass card. */}
      <header className="login-nav">
        <div className="login-brand">
          <span className="login-brand-mark">
            <Building2 className="size-5" aria-hidden="true" />
          </span>
          <span>Office Brain</span>
        </div>
        {onDemo && (
          <button
            type="button"
            onClick={onDemo}
            disabled={demoBusy}
            className="login-nav-demo"
          >
            {demoBusy ? "Starting demo..." : "Try the demo →"}
          </button>
        )}
      </header>

      {/* Two-column area: headline + subhead sit directly on the background's
       * calm zone, pushed off the hard left edge into center-left; the wide,
       * compact glass card holds the form at center-right. */}
      <main className="login-main">
        {(title || subtitle) && (
          <div className="login-hero">
            {title && <h1 className="login-headline">{title}</h1>}
            {subtitle && <p className="login-subtext">{subtitle}</p>}
          </div>
        )}
        <div className="login-glass">{children}</div>
      </main>

      {/* Small secondary brand mark, bottom-left. */}
      <div className="login-corner-mark" aria-hidden="true">
        <span className="login-corner-mark-box">
          <Building2 className="size-4" />
        </span>
      </div>
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
    <div className="login-field-group">
      <label className="login-label">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        autoComplete={autoComplete}
        required={required}
        className="login-input"
      />
      {hint && <p className="login-hint">{hint}</p>}
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
    <div className="login-field-group">
      <label className="login-label">{label}</label>
      <div className="login-password-wrap">
        <input
          type={showPassword ? "text" : "password"}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          autoComplete={autoComplete}
          required={required}
          className="login-input"
        />
        <button
          type="button"
          className="login-password-toggle"
          onClick={() => setShowPassword(!showPassword)}
          tabIndex={-1}
          aria-label={showPassword ? "Hide password" : "Show password"}
        >
          {showPassword ? (
            <EyeOff className="size-4" aria-hidden="true" />
          ) : (
            <Eye className="size-4" aria-hidden="true" />
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
    <div className="space-y-1.5 pt-1">
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
        <div
          className={cn(
            "h-full transition-all duration-300 ease-in-out",
            colorClass,
          )}
          style={{ width }}
        />
      </div>
      <p className="flex justify-between text-[11px] font-medium text-white/60">
        <span>Password strength:</span>
        <span
          className={cn(
            strength === "Weak" && "text-[#FF6B6B]",
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