"use client";

/**
 * Organization verification page.
 *
 * This page is reached via the verification link sent in the organization
 * verification email. It exchanges the token for a verified status.
 */

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

const BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

function VerifyOrganization() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [orgName, setOrgName] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    const run = async () => {
      const token = searchParams.get("token");
      if (!token) {
        if (active) {
          setStatus("error");
          setError("No verification token found. Please use the link from your email.");
        }
        return;
      }

      try {
        const response = await fetch(
          `${BASE_URL}/verify-organization?token=${encodeURIComponent(token)}`,
          {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
            },
          }
        );

        if (!active) return;

        if (response.ok) {
          const data = await response.json();
          setOrgName(data.name);
          setStatus("success");
        } else {
          const data = await response.json().catch(() => ({}));
          setStatus("error");
          setError(data.detail || "Invalid or expired verification link.");
        }
      } catch {
        if (active) {
          setStatus("error");
          setError("Could not reach the server. Please try again later.");
        }
      }
    };

    void run();
    return () => {
      active = false;
    };
  }, [searchParams]);

  if (status === "loading") {
    return (
      <div className="flex min-h-dvh flex-col items-center justify-center gap-4 bg-background">
        <Loader2 className="size-10 animate-spin text-accent" aria-hidden />
        <p className="text-sm text-muted">Verifying your organization…</p>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-background p-4">
        <div className="w-full max-w-md rounded-2xl border border-border bg-surface-raised p-8 text-center shadow-xl">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-danger/10">
            <AlertCircle className="h-6 w-6 text-danger" aria-hidden />
          </div>
          <h1 className="mb-2 text-xl font-semibold text-foreground">
            Verification failed
          </h1>
          <p className="mb-6 text-sm text-muted">{error}</p>
          <button
            type="button"
            onClick={() => router.replace("/login")}
            className={cn(
              "rounded-lg bg-accent px-4 py-2 text-sm font-medium text-accent-foreground",
              "transition-colors hover:bg-accent/90",
              "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none"
            )}
          >
            Back to sign in
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-dvh items-center justify-center bg-background p-4">
      <div className="w-full max-w-md rounded-2xl border border-border bg-surface-raised p-8 text-center shadow-xl">
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-accent/10">
          <CheckCircle className="h-6 w-6 text-accent" aria-hidden />
        </div>
        <h1 className="mb-2 text-xl font-semibold text-foreground">
          Organization verified
        </h1>
        <p className="mb-6 text-sm text-muted">
          {orgName ? (
            <>
              <strong>{orgName}</strong> has been successfully verified.
            </>
          ) : (
            "Your organization has been successfully verified."
          )}
        </p>
        <button
          type="button"
          onClick={() => router.replace("/")}
          className={cn(
            "rounded-lg bg-accent px-4 py-2 text-sm font-medium text-accent-foreground",
            "transition-colors hover:bg-accent/90",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none"
          )}
        >
          Go to application
        </button>
      </div>
    </div>
  );
}

export default function VerifyOrganizationPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-dvh items-center justify-center bg-background">
          <Loader2 className="size-10 animate-spin text-accent" aria-hidden />
        </div>
      }
    >
      <VerifyOrganization />
    </Suspense>
  );
}
