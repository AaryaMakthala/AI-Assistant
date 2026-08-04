'use client';

/**
 * Set a new password.
 *
 * Reached only via the emailed recovery link, which routes through `/auth/callback` and
 * leaves a session behind. That session is the authorization for the change — Supabase's
 * `updateUser` acts on the current user, so with no session there is nobody to update.
 * The gate below checks for it up front rather than letting the form submit into a raw
 * "Auth session missing" error.
 */

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Loader2, AlertCircle, CheckCircle2, Eye, EyeOff, LockKeyhole } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getSupabaseClient } from '@/lib/supabase/client';

const MIN_PASSWORD_LENGTH = 8;

type Gate = 'checking' | 'ready' | 'no-session';

export default function ResetPasswordPage() {
  const router = useRouter();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // An unconfigured deployment is known before any effect runs, so it is the initial state
  // rather than something an effect corrects after the first render.
  const [gate, setGate] = useState<Gate>(() =>
    getSupabaseClient() ? 'checking' : 'no-session',
  );

  useEffect(() => {
    const supabase = getSupabaseClient();
    if (!supabase) return;

    let active = true;
    // getSession() awaits the client's initialization, so this settles after any session
    // established by the recovery link has been stored.
    void supabase.auth.getSession().then(({ data }) => {
      if (active) setGate(data.session ? 'ready' : 'no-session');
    });
    return () => {
      active = false;
    };
  }, []);

  const strength = getPasswordStrength(password);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (password.length < MIN_PASSWORD_LENGTH) {
      setError(`Password must be at least ${MIN_PASSWORD_LENGTH} characters long.`);
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    const supabase = getSupabaseClient();
    if (!supabase) {
      setError('This deployment has no Supabase credentials.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const { error: updateError } = await supabase.auth.updateUser({ password });

      if (updateError) {
        setError(
          updateError.message.toLowerCase().includes('session')
            ? 'Your reset link has expired. Request a new one and try again.'
            : updateError.message,
        );
        return;
      }

      setIsSuccess(true);
      // Signed out deliberately: the new password should be proven by using it, and it
      // leaves the account in the same state on every device.
      await supabase.auth.signOut();
      setTimeout(() => router.replace('/login'), 2500);
    } catch {
      setError('Could not reach the authentication service. Please check your connection.');
    } finally {
      setIsLoading(false);
    }
  };

  if (gate === 'checking') {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-background">
        <Loader2 className="size-10 animate-spin text-accent" aria-hidden />
      </div>
    );
  }

  return (
    <div className="flex min-h-dvh items-center justify-center bg-background p-4">
      <div className="w-full max-w-md overflow-hidden rounded-2xl border border-border bg-surface-raised p-8 shadow-xl animate-fade-in">
        <div className="mb-6 flex flex-col items-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-accent-subtle">
            <LockKeyhole className="h-6 w-6 text-accent" aria-hidden />
          </div>
          <h1 className="text-center text-2xl font-semibold tracking-tight text-foreground">
            Set new password
          </h1>
          <p className="mt-2 text-center text-sm text-muted">
            {gate === 'no-session'
              ? 'This page can only be opened from a password reset email.'
              : 'Please enter your new password below.'}
          </p>
        </div>

        {gate === 'no-session' ? (
          <Link
            href="/forgot-password"
            className="flex w-full items-center justify-center rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-accent-foreground transition-colors hover:bg-accent/90"
          >
            Request a reset link
          </Link>
        ) : isSuccess ? (
          <div
            role="status"
            className="flex flex-col items-center rounded-xl border border-success/20 bg-success/10 p-6 text-center"
          >
            <CheckCircle2 className="mb-3 h-8 w-8 text-success" aria-hidden />
            <h2 className="mb-1 text-base font-semibold text-success">Password reset successfully</h2>
            <p className="text-sm text-success/80">Redirecting you to sign in…</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div
                role="alert"
                className="flex items-start rounded-lg border border-danger/20 bg-danger/10 p-3 text-sm text-danger"
              >
                <AlertCircle className="mr-2 mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                <span>{error}</span>
              </div>
            )}

            <div className="space-y-2">
              <label htmlFor="password" className="text-sm font-medium text-foreground">
                New password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  required
                  autoComplete="new-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="flex w-full rounded-lg border border-border bg-surface px-3 py-2 pr-10 text-sm text-foreground transition-colors placeholder:text-muted focus:border-accent focus:ring-1 focus:ring-accent focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  tabIndex={-1}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted transition-colors hover:text-foreground focus:outline-none"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>

              {password.length > 0 && (
                <div className="mt-2 flex gap-1" aria-hidden>
                  {[0, 1, 2, 3].map((index) => (
                    <div
                      key={index}
                      className={cn(
                        'h-1 w-full rounded-full transition-colors',
                        index < strength.score ? strength.color : 'bg-border',
                      )}
                    />
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-2 pb-2">
              <label htmlFor="confirmPassword" className="text-sm font-medium text-foreground">
                Confirm new password
              </label>
              <input
                id="confirmPassword"
                type={showPassword ? 'text' : 'password'}
                required
                autoComplete="new-password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="flex w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-foreground transition-colors placeholder:text-muted focus:border-accent focus:ring-1 focus:ring-accent focus:outline-none"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading || !password || !confirmPassword}
              className={cn(
                'flex w-full items-center justify-center rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-accent-foreground transition-colors hover:bg-accent/90 focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none',
                (isLoading || !password || !confirmPassword) && 'cursor-not-allowed opacity-70',
              )}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />
                  Updating password…
                </>
              ) : (
                'Reset password'
              )}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

function getPasswordStrength(password: string): { score: number; color: string } {
  let score = 0;
  if (password.length >= MIN_PASSWORD_LENGTH) score += 1;
  if (/[A-Z]/.test(password) && /[a-z]/.test(password)) score += 1;
  if (/[0-9]/.test(password)) score += 1;
  if (/[^A-Za-z0-9]/.test(password)) score += 1;

  if (score <= 1) return { score, color: 'bg-danger' };
  if (score === 2) return { score, color: 'bg-warning' };
  if (score === 3) return { score, color: 'bg-success/70' };
  return { score, color: 'bg-success' };
}
