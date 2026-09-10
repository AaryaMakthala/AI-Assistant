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
  children,
}: {
  title?: string;
  subtitle?: string;
  onDemo?: () => void;
  demoBusy?: boolean;
  /** Use a wider glass card. Signup mode pairs this with a two-column field
   *  grid so the form reads as a wide horizontal panel, not a tall stack. */
  cardWide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="auth-page">
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
                    <span style={{ color: "#8FAE83" }}>Office Brain</span>
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
    </div>
  );
}
