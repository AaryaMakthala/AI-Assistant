// CDP probe: verify the redesigned chat workspace.
//   - chat-bg.png loads (200) and is the full-bleed background
//   - backdrop-filter matrix: sidebar/top bar/composer have glass,
//     the reading area has NONE (legibility requirement)
//   - fonts (Fraunces headings), pill New chat button, sidebar sections
//   - no scrollbars at any viewport; screenshots captured
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const DEBUG_PORT = 9334;
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
    "--user-data-dir=C:/Temp/cdp-chat-profile",
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
      // Let React hooks settle and the workspace render.
      await new Promise((r) => setTimeout(r, 2000));

      const statuses = new Map();
      for (const e of cdp.events.slice(before)) {
        if (e.method === "Network.responseReceived") {
          const u = e.params.response.url || "";
          if (u.includes("chat-bg.png") || u.includes("/blobs/")) {
            statuses.set(u.split("?")[0].replace(/^https?:\/\/[^/]+/, ""), e.params.response.status);
          }
        }
      }

      const expr = `(() => {
        const q = (s) => document.querySelector(s);
        const all = (s) => [...document.querySelectorAll(s)];
        const rect = (el) => { const r = el.getBoundingClientRect(); return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) }; };
        const css = (el, p) => el ? getComputedStyle(el)[p] : null;
        const bgImg = all("img").find((img) => img.src.includes("chat-bg")) || null;
        const sidebar = q("aside");
        const header = all("header").find((h) => h.classList.contains("glass-bar")) || null;
        const composerBox = q(".glass-input");
        const scrollArea = q(".chat-scroll");
        const h1 = q("main h1");
        const newChat = all("aside button").find((b) => b.textContent && b.textContent.includes("New chat")) || null;
        const tabs = all("aside [role=tablist] button").map((b) => b.textContent.trim());
        const emptyH2 = all("main h2").find((h) => h.textContent.includes("Ask about anything"));
        const suggestionBtns = all("main button").filter((b) => b.textContent && b.textContent.includes("?"));

        return {
          bgImgSrc: bgImg ? bgImg.src.slice(0, 130) : null,
          bgLoaded: bgImg ? bgImg.complete && bgImg.naturalWidth > 0 : false,
          bgNatural: bgImg ? { w: bgImg.naturalWidth, h: bgImg.naturalHeight, aspect: (bgImg.naturalWidth / bgImg.naturalHeight).toFixed(3) } : null,
          bgObjectFit: bgImg ? css(bgImg, "objectFit") : null,
          bgRect: bgImg ? rect(bgImg) : null,
          sidebar: sidebar ? { backdrop: css(sidebar, "backdropFilter"), bg: css(sidebar, "backgroundColor"), radius: css(sidebar, "borderRadius"), borderRight: css(sidebar, "borderRightWidth") + " " + css(sidebar, "borderRightColor") } : null,
          header: header ? { backdrop: css(header, "backdropFilter"), bg: css(header, "backgroundColor"), borderBottom: css(header, "borderBottomWidth") + " " + css(header, "borderBottomColor") } : null,
          composer: composerBox ? { backdrop: css(composerBox, "backdropFilter"), bg: css(composerBox, "backgroundColor"), radius: css(composerBox, "borderRadius"), border: css(composerBox, "borderTopWidth") + " " + css(composerBox, "borderTopColor"), rect: rect(composerBox) } : null,
          readingArea: scrollArea ? { backdrop: css(scrollArea, "backdropFilter"), bg: css(scrollArea, "backgroundColor"), rect: rect(scrollArea) } : null,
          h1: { font: h1 ? css(h1, "fontFamily").slice(0, 50) : null, size: h1 ? css(h1, "fontSize") : null },
          emptyH2: { font: emptyH2 ? css(emptyH2, "fontFamily").slice(0, 50) : null, size: emptyH2 ? css(emptyH2, "fontSize") : null },
          newChatPill: newChat ? { radius: css(newChat, "borderRadius"), bg: css(newChat, "backgroundColor"), color: css(newChat, "color") } : null,
          suggestionCard: suggestionBtns[0] ? { bg: css(suggestionBtns[0], "backgroundColor"), border: css(suggestionBtns[0], "borderTopColor"), radius: css(suggestionBtns[0], "borderRadius") } : null,
          suggestionCount: suggestionBtns.length,
          tabs,
          emptyStateShown: emptyH2 !== null,
          bodyBg: css(document.body, "backgroundColor"),
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
    }

    console.log(JSON.stringify(metrics, null, 1));
  } finally {
    chrome.kill();
  }
}

main().catch((e) => { console.error("FATAL", e); process.exit(1); });