'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Loader2, AlertCircle, CheckCircle2, KeyRound } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getSupabaseClient } from '@/lib/supabase/client';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;

    const supabase = getSupabaseClient();
    if (!supabase) {
      setError('Sign-in is unavailable: this deployment has no Supabase credentials.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setIsSuccess(false);

    try {
      const { error: resetError } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/auth/callback?next=/reset-password`,
      });

      // Deliberately not surfaced: whether an address is registered is not something an
      // unauthenticated visitor should be able to probe. The same confirmation shows either
      // way, so a failure here is logged rather than shown.
      if (resetError) console.warn('Password reset request failed', resetError);

      setIsSuccess(true);
    } catch {
      setError('Could not reach the authentication service. Please check your connection.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="w-full max-w-md overflow-hidden rounded-2xl border border-border bg-surface-raised p-8 shadow-xl animate-fade-in">
        <div className="flex flex-col items-center mb-6">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-accent-subtle">
            <KeyRound className="h-6 w-6 text-accent" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-foreground text-center">
            Forgot password
          </h1>
          <p className="mt-2 text-sm text-muted text-center">
            Enter your email address and we&apos;ll send you a link to reset your password.
          </p>
        </div>

        {error && (
          <div className="mb-6 rounded-lg bg-danger/10 p-3 text-sm text-danger flex items-start">
            <AlertCircle className="mr-2 h-4 w-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {isSuccess ? (
          <div className="flex flex-col items-center rounded-xl bg-success/10 p-6 text-center border border-success/20 mb-6">
            <CheckCircle2 className="mb-3 h-8 w-8 text-success" />
            <p className="text-sm font-medium text-success">
              If an account exists for that email, we&apos;ve sent a password reset link. Check your inbox.
            </p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4 mb-6">
            <div className="space-y-2">
              <label htmlFor="email" className="text-sm font-medium text-foreground">
                Email address
              </label>
              <input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@example.com"
                className="flex w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-foreground transition-colors placeholder:text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading || !email}
              className={cn(
                "flex w-full items-center justify-center rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-accent-foreground transition-colors hover:bg-accent/90 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-background",
                (isLoading || !email) && "opacity-70 cursor-not-allowed"
              )}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Sending link...
                </>
              ) : (
                "Send reset link"
              )}
            </button>
          </form>
        )}

        <div className="flex justify-center">
          <Link 
            href="/login"
            className="flex items-center text-sm font-medium text-muted hover:text-foreground transition-colors"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to sign in
          </Link>
        </div>
      </div>
    </div>
  );
}
