"use client";

/**
 * Shared mapping of Supabase Auth errors to application error codes and
 * user-facing messages.
 *
 * The three auth flows that surface Supabase errors directly — sign-up, email
 * resend, and password reset — call `mapSupabaseAuthError(error)` and branch
 * on the returned application code:
 *
 *   const [errorCode, userMessage] = mapSupabaseAuthError(error);
 *   if (errorCode === "EMAIL_LIMIT_EXCEEDED") { showToast(userMessage); }
 *   else { setError(userMessage); }
 *
 * Email-sending rate limits are shown as a toast (they are operational, not
 * form errors), everything else renders inline. This module owns the mapping
 * so all three pages stay in step; the pages never read `error.message`
 * through this path.
 *
 * Supabase error codes are matched first (stable, machine-readable), then the
 * HTTP status, then the message text as a last resort — some transports
 * surface rate limits without a populated `code`.
 */

import type { AuthError } from "@supabase/supabase-js";

/** Application error codes returned as the first element of the tuple. */
export type SupabaseAuthErrorCode =
  | "EMAIL_LIMIT_EXCEEDED"
  | "EMAIL_NOT_AUTHORIZED"
  | "USER_ALREADY_EXISTS"
  | "AUTH_ERROR";

/** User-facing message for the email-send rate limit (code or HTTP 429). */
const EMAIL_LIMIT_MESSAGE =
  "Email sending limit reached. The email was not sent. Please try again later or upgrade your Supabase plan.";

/** User-facing message when the error cannot be mapped to anything specific. */
const GENERIC_AUTH_MESSAGE = "Authentication failed. Please try again.";

/** Map a Supabase Auth error to an application code and a user-facing message.
 *
 * Null/undefined or unrecognized errors fall back to a generic entry rather
 * than throwing — callers destructure the result unconditionally.
 */
export function mapSupabaseAuthError(
  error: AuthError | null | undefined,
): [SupabaseAuthErrorCode, string] {
  if (!error) return ["AUTH_ERROR", GENERIC_AUTH_MESSAGE];

  const code = typeof error.code === "string" ? error.code : "";
  const message = typeof error.message === "string" ? error.message : "";
  const status = typeof error.status === "number" ? error.status : undefined;

  // Email sending rate limit: explicit Supabase code, or the corresponding
  // HTTP 429 response, or rate-limit wording when neither is populated.
  if (
    code === "over_email_send_rate_limit" ||
    status === 429 ||
    /rate limit|too many requests|\b429\b/i.test(message)
  ) {
    return ["EMAIL_LIMIT_EXCEEDED", EMAIL_LIMIT_MESSAGE];
  }

  // The deployment's email sender address is not authorized in Supabase.
  if (code === "email_address_not_authorized") {
    return [
      "EMAIL_NOT_AUTHORIZED",
      "This deployment cannot send email to that address. Please contact your administrator.",
    ];
  }

  // Duplicate sign-up: Supabase uses different codes for the two transports.
  if (code === "user_already_exists" || code === "email_exists") {
    return [
      "USER_ALREADY_EXISTS",
      "An account with this email address already exists. Please sign in instead.",
    ];
  }

  // Unknown error: surface Supabase's own message when it is usable so the
  // caller's inline error shows something truthful; otherwise stay generic.
  return ["AUTH_ERROR", message.trim() ? message : GENERIC_AUTH_MESSAGE];
}
