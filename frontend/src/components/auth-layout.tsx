"use client";

/**
 * Shared authentication layout — full-bleed background image, decorative blob
 * accents, top nav bar, and two-column hero + glass card arrangement. Used by
 * login, signup, and other auth-related pages so they all share the same visual
 * treatment without drift.
 *
 * The component is a pure presentational shell: it accepts title, subtitle,
 * optional nav actions, and children (the form content). All auth state
 * management stays in the individual page components.
 */

import Image from "next/image";
import { BrandWordmark } from "@/components/brand-wordmark";
import { cn } from "@/lib/utils";
import { Mail } from "lucide-react";

/** Decorative blob accents — small PNGs from public/blobs/, individually
 * positioned above the background but behind the nav + glass card. The
 * large stone blob is the primary decorative element; green is a small
 * accent. */
const BLOBS = [
  {
    id: "gcircle",
    src: "/blobs/stone.png",
    width: 677,
    height: 369,
    className: "auth-blob auth-blob-gcircle",
  },
  {
    id: "green",
    src: "/blobs/green.png",
    width: 56,
    height: 56,
    className: "auth-blob auth-blob-green",
  },
] as const;

export function AuthLayout({
  title,
  subtitle,
  onDemo,
  demoBusy,
  cardWide = false,
  className,
  children,
}: {
  title?: string;
  subtitle?: string;
  onDemo?: () => void;
  demoBusy?: boolean;
  /** Use a wider glass card. Signup mode pairs this with a two-column field
   *  grid so the form reads as a wide horizontal panel, not a tall stack. */
  cardWide?: boolean;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("auth-page", className)}>
      {/* Full-bleed background: one next/image layer of login-bg-5K.png,
       * object-fit: cover — fills the entire viewport edge-to-edge with zero
       * letterboxing. The photo's calm dark zone on the left is what the
       * headline sits on; cover's minor edge-crop never reaches it. */}
      <Image
        src="/login-bg-5K.png"
        alt=""
        aria-hidden
        fill
        priority
        unoptimized
        sizes="100vw"
        draggable={false}
        style={{ objectFit: "cover", objectPosition: "center" }}
        className="auth-bg"
      />

      {/* SVG noise filter for dithering blob gradients — hidden, referenced by CSS. */}
      <svg className="sr-only" aria-hidden="true">
        <filter id="blob-noise">
          <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
      </svg>

      {/* Decorative blob layer — above background, below nav + card. */}
      <div className="auth-blobs" aria-hidden="true">
        {BLOBS.map((blob) => (
          <Image
            key={blob.id}
            src={blob.src}
            alt=""
            width={blob.width}
            height={blob.height}
            unoptimized
            priority
            className={blob.className}
          />
        ))}
      </div>

      {/* Top nav — brand mark left, optional demo button right. */}
      <header className="auth-nav">
        <BrandWordmark
          withMark
          className="auth-brand"
          textClassName="font-sans text-[11px] font-semibold uppercase text-[#9CB88F] tracking-[0.08em]"
        />
        {onDemo && (
          <button
            type="button"
            onClick={onDemo}
            disabled={demoBusy}
            className="auth-nav-demo"
          >
            {demoBusy ? "Starting demo..." : "Try the demo →"}
          </button>
        )}
      </header>

      {/* Two-column area: headline + subhead on the left, glass card on right. */}
      <main className="auth-main">
        {(title || subtitle) && (
          <div className="auth-hero">
            {title && (
              <h1 className="auth-headline">
                {title.includes("Office Brain") ? (
                  <>
                    Sign in to{" "}
                    <span style={{ color: "#1e5a3a" }}>Office Brain</span>
                  </>
                ) : (
                  title
                )}
              </h1>
            )}
            {subtitle && <p className="auth-subtext">{subtitle}</p>}
          </div>
        )}
        <div className={cn("auth-glass", cardWide && "auth-glass-wide")}>{children}</div>
      </main>

      <footer className="auth-footer">
        <p className="auth-footer-copy">
          &copy; 2026 Makthala Aarya. All rights reserved.
        </p>
        <div className="auth-footer-links">
          <a
            href="https://www.linkedin.com/in/aaryamakthala/"
            target="_blank"
            rel="noopener noreferrer"
            className="auth-footer-pill"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
              <rect width="4" height="12" x="2" y="9" />
              <circle cx="4" cy="4" r="2" />
            </svg>
            LinkedIn
          </a>
          <a
            href="https://github.com/AaryaMakthala"
            target="_blank"
            rel="noopener noreferrer"
            className="auth-footer-pill"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
              <path d="M9 18c-4.51 2-5-2-7-2" />
            </svg>
            GitHub
          </a>
          <a
            href="mailto:aaryamakthala@gmail.com"
            className="auth-footer-pill"
          >
            <Mail />
            Email
          </a>
        </div>
      </footer>
    </div>
  );
}
