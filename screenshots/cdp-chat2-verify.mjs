// CDP probe: verify the Claude/ChatGPT-style chat restructure.
//   - sidebar: solid color, backdrop-filter NONE, no image
//   - chat-bg.png: present ONLY inside the message frame (rect check), 200
//   - glass message cards render (computed styles) with legible text over the image
//   - centered column, clean composer, no scrollbars; screenshots captured
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const DEBUG_PORT = 9335;
const BASE = "http://localhost:3000/";

const VPs = [
  { name: "desktop900", width: 1440, height: 900 },
  { name: "desktop800", width: 1440, height: 800 },
  { name: "mobile844", width: 390, height: 844 },
];

function launchChrome() {
  return spawn(CHROME, [
    `--remote-debugging-port=${DEBUG_PORT}`,
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--user-data-dir=C:/Temp/cdp-chat2-profile",
    "about:blank",
  ], { stdio: "ignore" });
}

async function getPageTarget() {
  for (let i = 0; i < 40; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
      const targets = await res.json();
      const page = targets.find((t) => t.type === "page");
      if (page) return page;
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("no chrome page target");
}

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.events = [];
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id) {
        const p = this.pending.get(msg.id);
        if (p) {
          this.pending.delete(msg.id);
          msg.error ? p.reject(new Error(JSON.stringify(msg.error))) : p.resolve(msg.result);
        }
      } else {
        this.events.push(msg);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((res, rej) => { ws.addEventListener("open", res); ws.addEventListener("error", rej); });
    return new CDP(ws);
  }
}

async function main() {
  const chrome = launchChrome();
  try {
    const target = await getPageTarget();
    const cdp = await CDP.connect(target.webSocketDebuggerUrl);

    await cdp.send("Page.enable");
    await cdp.send("Network.enable");
    await cdp.send("Emulation.setEmulatedMedia", {
      features: [{ name: "prefers-reduced-motion", value: "reduce" }],
    });

    const metrics = [];
    for (const vp of VPs) {
      await cdp.send("Emulation.setDeviceMetricsOverride", {
        width: vp.width,
        height: vp.height,
        deviceScaleFactor: 1,
        mobile: vp.width < 768,
      });
      const before = cdp.events.length;
      await cdp.send("Page.navigate", { url: BASE });
      for (let i = 0; i < 60; i++) {
        const loaded = cdp.events.slice(before).some((e) => e.method === "Page.loadEventFired");
        if (loaded) break;
        await new Promise((r) => setTimeout(r, 200));
      }
      await new Promise((r) => setTimeout(r, 2000));

      const statuses = new Map();
      for (const e of cdp.events.slice(before)) {
        if (e.method === "Network.responseReceived") {
          const u = e.params.response.url || "";
          if (u.includes("chat-bg.png")) {
            statuses.set(u.split("?")[0].replace(/^https?:\/\/[^/]+/, ""), e.params.response.status);
          }
        }
      }

      const expr = `(() => {
        const q = (s) => document.querySelector(s);
        const all = (s) => [...document.querySelectorAll(s)];
        const rect = (el) => { const r = el.getBoundingClientRect(); return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), right: Math.round(r.right), bottom: Math.round(r.bottom) }; };
        const css = (el, p) => el ? getComputedStyle(el)[p] : null;
        const sidebar = q("aside");
        const header = all("header").find((h) => h.textContent.includes("Knowledge Assistant")) || null;
        const chatScroll = q(".chat-scroll");
        const bgImg = all("img").find((img) => img.src.includes("chat-bg")) || null;
        const composerBox = document.querySelector("textarea")?.parentElement ?? null;
        const col = chatScroll ? chatScroll.firstElementChild : null;
        const h1 = q("main h1");
        return {
          sidebar: sidebar ? { backdrop: css(sidebar, "backdropFilter"), bg: css(sidebar, "backgroundColor"), backgroundImage: css(sidebar, "backgroundImage") } : null,
          header: header ? { backdrop: css(header, "backdropFilter"), bg: css(header, "backgroundColor") } : null,
          bgImg: bgImg ? { src: bgImg.src.slice(0, 120), loaded: bgImg.complete && bgImg.naturalWidth > 0, aspect: (bgImg.naturalWidth / bgImg.naturalHeight).toFixed(3), fit: css(bgImg, "objectFit"), rect: rect(bgImg) } : null,
          chatScrollRect: chatScroll ? rect(chatScroll) : null,
          columnRect: col ? rect(col) : null,
          frameImgs: (() => {
            // every img whose rect intersects the scroll frame — should be exactly the bg
            const fr = chatScroll ? chatScroll.getBoundingClientRect() : null;
            if (!fr) return null;
            return all("img").filter((img) => {
              const r = img.getBoundingClientRect();
              return r.left >= fr.left - 1 && r.right <= fr.right + 1;
            }).map((img) => img.src.slice(0, 80));
          })(),
          composer: composerBox ? { bg: css(composerBox, "backgroundColor"), border: css(composerBox, "borderTopColor"), radius: css(composerBox, "borderRadius"), rect: rect(composerBox) } : null,
          composerFound: composerBox !== null,
          newChatBtn: (() => {
            const b = all("aside button").find((x) => x.textContent.includes("New chat"));
            return b ? { bg: css(b, "backgroundColor"), color: css(b, "color"), radius: css(b, "borderRadius") } : null;
          })(),
          emptyStateShown: document.body.textContent.includes("Ask about anything your organization knows"),
          scroll: {
            sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth,
            sh: document.documentElement.scrollHeight, ch: document.documentElement.clientHeight,
          },
        };
      })()`;
      const res = await cdp.send("Runtime.evaluate", { expression: expr, returnByValue: true });
      if (res.exceptionDetails) throw new Error("evaluate: " + JSON.stringify(res.exceptionDetails));
      const m = res.result.value;
      m.statuses = Object.fromEntries(statuses);
      metrics.push({ name: vp.name, ...m });

      const shot = await cdp.send("Page.captureScreenshot", { format: "png" });
      mkdirSync("screenshots", { recursive: true });
      writeFileSync(`screenshots/chat-${vp.name}.png`, Buffer.from(shot.data, "base64"));

      // On desktop900 only: inject a REAL assistant message card (same classes the
      // component emits) so glass-card legibility can be measured over the photo.
      if (vp.name === "desktop900") {
        await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
        const inject = `(() => {
          const col = document.querySelector(".chat-scroll > div");
          const wrap = document.createElement("div");
          wrap.innerHTML = \`<div class="space-y-6">
            <div class="flex justify-end gap-3"><div class="max-w-[min(34rem,80%)] rounded-[18px] rounded-br-[6px] bg-accent px-[18px] py-3 text-sm font-medium leading-relaxed whitespace-pre-wrap text-accent-foreground shadow-[0_2px_8px_rgba(0,0,0,0.15)]">How much PTO do I get and can I carry it over?</div><div class="mt-1 flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-raised text-muted"><svg class="size-3.5" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\"><path d=\"M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2\"/><circle cx=\"12\" cy=\"7\" r=\"4\"/></svg></div></div>
            <div class="flex gap-3"><div class="mt-1 flex size-7 shrink-0 items-center justify-center rounded-full border border-[rgba(255,255,255,0.12)] bg-[#0F1A15] text-xs font-semibold text-[rgba(245,243,236,0.85)]" aria-hidden="true">AI</div><div class="min-w-0 flex-1"><div class="glass-message p-5"><div class="space-y-3"><div><p class="text-sm leading-relaxed">Paid time off accrues at 1.67 days per month, which works out to 20 days per calendar year for full-time employees, and you can carry up to five unused days into the following year. Requests over five days need manager approval in advance.</p></div></div></div></div></div>
          </div>\`;
          col.prepend(wrap);
          return true;
        })()`;
        await cdp.send("Runtime.evaluate", { expression: inject, returnByValue: true });
        await new Promise((r) => setTimeout(r, 800));
        const shot2 = await cdp.send("Page.captureScreenshot", { format: "png" });
        writeFileSync("screenshots/chat-desktop900-messages.png", Buffer.from(shot2.data, "base64"));
        const cleanup = await cdp.send("Runtime.evaluate", {
          expression: `(() => { const c = document.querySelector(".chat-scroll > div"); c.innerHTML = ""; const e = document.createElement("div"); e.className = ""; })()`,
          returnByValue: true,
        });
      }
    }

    console.log(JSON.stringify(metrics, null, 1));
  } finally {
    chrome.kill();
  }
}

main().catch((e) => { console.error("FATAL", e); process.exit(1); });