'use client';

import { Suspense, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Mail, ArrowLeft, Loader2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getSupabaseClient } from '@/lib/supabase/client';

function CheckEmailContent() {
  const searchParams = useSearchParams();
  const email = searchParams.get('email');
  
  const [isResending, setIsResending] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleResend = async () => {
    if (!email) return;
    
    const supabase = getSupabaseClient();
    if (!supabase) {
      setError('Sign-in is unavailable: this deployment has no Supabase credentials.');
      return;
    }

    setIsResending(true);
    setError(null);
    setResendSuccess(false);

    try {
      const { error: resendError } = await supabase.auth.resend({
        type: 'signup',
        email,
        options: { emailRedirectTo: `${window.location.origin}/auth/callback` },
      });

      if (resendError) {
        setError(resendError.message);
        return;
      }

      setResendSuccess(true);
    } catch {
      setError('Could not reach the authentication service. Please check your connection.');
    } finally {
      setIsResending(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="w-full max-w-md overflow-hidden rounded-2xl border border-border bg-surface-raised p-8 shadow-xl animate-fade-in">
        <div className="flex flex-col items-center text-center">
          <div className="relative mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-accent-subtle">
            <Mail className="h-10 w-10 text-accent animate-pulse" />
          </div>
          
          <h1 className="mb-2 text-2xl font-semibold tracking-tight text-foreground">
            Check your email
          </h1>
          
          <p className="mb-8 text-sm text-muted">
            {email ? (
              <>
                We&apos;ve sent a verification link to <span className="font-medium text-foreground">{email}</span>.
                Please click the link in the email to verify your account.
              </>
            ) : (
              "We've sent a verification link to your email."
            )}
          </p>

          {error && (
            <div className="mb-6 w-full rounded-lg bg-danger/10 p-3 text-sm text-danger flex items-start text-left">
              <AlertCircle className="mr-2 h-4 w-4 mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {resendSuccess && (
            <div className="mb-6 w-full rounded-lg bg-success/10 p-3 text-sm text-success text-center">
              Verification email sent successfully!
            </div>
          )}

          <div className="flex w-full flex-col space-y-4">
            {email && (
              <button
                onClick={handleResend}
                disabled={isResending}
                className={cn(
                  "flex w-full items-center justify-center rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-accent-foreground transition-colors hover:bg-accent/90 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-background",
                  isResending && "opacity-70 cursor-not-allowed"
                )}
              >
                {isResending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Resending...
                  </>
                ) : (
                  "Resend verification email"
                )}
              </button>
            )}
            
            <Link 
              href="/login"
              className="flex items-center justify-center text-sm font-medium text-muted hover:text-foreground transition-colors"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to sign in
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function CheckEmailPage() {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-accent" />
      </div>
    }>
      <CheckEmailContent />
    </Suspense>
  );
}
