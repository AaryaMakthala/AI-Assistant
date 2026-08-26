"use client";

/**
 * Markdown rendering for assistant answers.
 *
 * `react-markdown` parses to a React element tree and never touches `innerHTML`, so a
 * document that contains `<script>` renders as text rather than executing. That matters
 * more here than in most markdown surfaces: this content is model output written over
 * retrieved document text, which CLAUDE.md 4.4 treats as attacker-influenceable. No
 * `rehype-raw` — enabling raw HTML would give back exactly the injection surface the
 * parser is avoiding.
 */

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

const COMPONENTS: Components = {
  p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,
  ul: ({ children }) => (
    <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>
  ),
  li: ({ children }) => <li className="pl-0.5">{children}</li>,
  h1: ({ children }) => (
    <h1 className="mt-4 mb-2 text-base font-semibold first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="mt-4 mb-2 text-base font-semibold first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-3 mb-1.5 text-sm font-semibold first:mt-0">{children}</h3>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold">{children}</strong>
  ),
  code: ({ className, children }) => {
    // react-markdown distinguishes fenced blocks by the language class it sets; an
    // inline span has none.
    const isBlock = Boolean(className?.startsWith("language-"));
    if (isBlock) {
      return (
        <code className="font-mono text-[0.8125rem] leading-relaxed">
          {children}
        </code>
      );
    }
    return (
      <code className="rounded bg-surface-raised px-1 py-0.5 font-mono text-[0.8125rem]">
        {children}
      </code>
    );
  },
  pre: ({ children }) => (
    <pre className="my-3 overflow-x-auto rounded-lg border border-border bg-surface-raised p-3">
      {children}
    </pre>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-border pl-3 text-muted">
      {children}
    </blockquote>
  ),
  a: ({ children, href }) => (
    // Model output may contain a link drawn from an untrusted document. `noreferrer`
    // withholds the referrer, and `noopener` denies the opened page a handle back to
    // this window.
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer nofollow"
      className="text-accent underline underline-offset-2"
    >
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="my-3 overflow-x-auto">
      <table className="w-full border-collapse text-left text-[0.8125rem]">
        {children}
      </table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-border bg-surface-raised px-2 py-1 font-semibold">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-border px-2 py-1 align-top">{children}</td>
  ),
  hr: () => <hr className="my-4 border-border" />,
};

export function Markdown({
  content,
  className,
}: {
  content: string;
  className?: string;
}) {
  return (
    <div className={cn("text-sm leading-relaxed", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
