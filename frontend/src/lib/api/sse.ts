/**
 * SSE parsing for the chat stream.
 *
 * Hand-rolled rather than using `EventSource`: that API is GET-only and cannot carry an
 * Authorization header, and this endpoint is an authenticated POST. The parser is small
 * because the backend emits a strict subset of SSE — one `event:` line, one `data:` line,
 * blank-line terminated — but it still buffers across chunk boundaries, since a network
 * read can split a frame anywhere, including mid-UTF-8-character.
 */

import type { ChatStreamEvent } from "./types";

/** Frame names the backend emits. Anything else is ignored rather than thrown on, so
 * adding a frame server-side cannot break an older client. */
const KNOWN_EVENTS = new Set([
  "session",
  "routing",
  "step",
  "sources",
  "token",
  "citations",
  "done",
  "error",
]);

function parseFrame(frame: string): ChatStreamEvent | null {
  let name = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue; // comment / keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    // Per spec a single leading space after the colon is part of the framing.
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") name = value;
    else if (field === "data") dataLines.push(value);
  }

  if (!dataLines.length || !KNOWN_EVENTS.has(name)) return null;

  try {
    const payload = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
    return { ...payload, type: name } as ChatStreamEvent;
  } catch {
    // A truncated or malformed frame is dropped rather than aborting the stream: the
    // tokens already delivered are still valid, and the terminal frame may yet arrive.
    return null;
  }
}

/** Decode a byte stream into typed chat events, yielding each as it completes. */
export async function* parseEventStream(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  const reader = body.getReader();
  // `stream: true` keeps partial multi-byte sequences buffered across reads.
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) return;
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      // Normalize CRLF so the blank-line split works behind proxies that rewrite endings.
      buffer = buffer.replace(/\r\n/g, "\n");

      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseFrame(frame);
        if (event) yield event;
      }
    }

    // A final frame with no trailing blank line — a stream cut short at exactly the
    // wrong moment still gets one last chance to deliver a terminal event.
    const tail = (buffer + decoder.decode()).trim();
    if (tail) {
      const event = parseFrame(tail);
      if (event) yield event;
    }
  } finally {
    // Releasing the lock lets the underlying connection be torn down promptly when the
    // consumer stops iterating early (component unmount, user navigates away).
    reader.releaseLock();
  }
}
