// CDP probe: verify login page background (object-fit: contain, full photo),
// three blob imgs load 200 with float animations, asymmetric positions around
// the hero text, no scrollbars. Captures screenshots at three viewports.
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const DEBUG_PORT = 9333;
const BASE = "http://localhost:3000/login";

const VPs = [
  { name: "desktop900", width: 1440, height: 900 },
  { name: "desktop800", width: 1440, height: 800 },
  { name: "laptop1280", width: 1280, height: 800 },
  { name: "tablet1000", width: 1000, height: 800 },
  { name: "mobile844", width: 390, height: 844 },
];

// Opaque-shape geometry measured from the source PNGs (fraction of the element box).
const SHAPES = {
  "login-blob-gcircle": { x0: 204 / 677, w: (473 - 204) / 677, y0: 53 / 369, h: (319 - 53) / 369 },
  "login-blob-green": { x0: 89 / 500, w: (397 - 89) / 500, y0: 97 / 500, h: (406 - 97) / 500 },
  "login-blob-stone": { x0: 160 / 677, w: (509 - 160) / 677, y0: 48 / 369, h: (307 - 48) / 369 },
};

function launchChrome() {
  const proc = spawn(CHROME, [
    `--remote-debugging-port=${DEBUG_PORT}`,
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--user-data-dir=C:/Temp/cdp-login-profile",
    "about:blank",
  ], { stdio: "ignore" });
  return proc;
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
    // Freeze drift + entrance animations so metrics/screenshots are deterministic.
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
      // Wait for load
      for (let i = 0; i < 60; i++) {
        const loaded = cdp.events.slice(before).some((e) => e.method === "Page.loadEventFired");
        if (loaded) break;
        await new Promise((r) => setTimeout(r, 200));
      }
      await new Promise((r) => setTimeout(r, 600)); // let reduced-motion styles settle

      // collect network statuses observed since this navigation started
      const statuses = new Map();
      for (const e of cdp.events.slice(before)) {
        if (e.method === "Network.responseReceived") {
          const u = e.params.response.url || "";
          if (u.includes("login-bg.png") || u.includes("/blobs/")) {
            statuses.set(u.split("?")[0].replace(/^https?:\/\/[^/]+/, ""), e.params.response.status);
          }
        }
      }

      const expr = `(() => {
        const q = (s) => document.querySelector(s);
        const rect = (el) => { const r = el.getBoundingClientRect(); return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), right: Math.round(r.right), bottom: Math.round(r.bottom) }; };
        const bg = q(".login-bg");
        const hero = q(".login-hero");
        const glass = q(".login-glass");
        const nav = q(".login-nav");
        const blobs = {};
        for (const cls of ["gcircle", "green", "stone"]) {
          const el = q(".login-blob-" + cls);
          const r = el.getBoundingClientRect();
          blobs[cls] = {
            el: rect(el),
            anim: getComputedStyle(el).animationName,
            // visible-shape geometry derived from measured canvas fractions
            shape: (() => { const s = ${JSON.stringify(SHAPES)}["login-blob-" + cls];
              return { x: Math.round(r.x + s.x0 * r.width), y: Math.round(r.y + s.y0 * r.height), w: Math.round(s.w * r.width), h: Math.round(s.h * r.height) };
            })(),
          };
        }
        const headline = q(".login-headline") ? rect(q(".login-headline")) : null;
        const subtext = q(".login-subtext") ? rect(q(".login-subtext")) : null;
        // minimum gap (px) between each blob's visible shape and the card rect (0 = touching/overlapping)
        const cardGap = {};
        for (const cls of ["gcircle", "green", "stone"]) {
          const el = q(".login-blob-" + cls);
          const r = el.getBoundingClientRect();
          const s = ${JSON.stringify(SHAPES)}["login-blob-" + cls];
          const gx = r.x + s.x0 * r.width, gy = r.y + s.y0 * r.height;
          const gw = s.w * r.width, gh = s.h * r.height;
          const c = glass.getBoundingClientRect();
          const dx = Math.max(c.x - (gx + gw), gx - (c.x + c.width), 0);
          const dy = Math.max(c.y - (gy + gh), gy - (c.y + c.height), 0);
          cardGap[cls] = Math.round(Math.hypot(dx, dy));
        }
        const page = q(".login-page");
        const pageStyle = getComputedStyle(page);
        const input = q(".login-input");
        const inputStyle = getComputedStyle(input);
        const glassStyle = getComputedStyle(glass);
        const inputRect = input ? rect(input) : null;
        return {
          inputRect,
          cardGap,
          inputStyle: { borderColor: inputStyle.borderTopColor, background: inputStyle.backgroundColor },
          glassStyle: { background: glassStyle.backgroundColor },
          pageStyle: { backgroundImage: pageStyle.backgroundImage, backgroundColor: pageStyle.backgroundColor },
          objectFit: getComputedStyle(bg).objectFit,
          natural: { w: bg.naturalWidth, h: bg.naturalHeight, aspect: (bg.naturalWidth / bg.naturalHeight).toFixed(3) },
          bgRect: rect(bg),
          nav: rect(nav), hero: hero ? rect(hero) : null, glass: rect(glass),
          headline, subtext,
          blobs,
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

      // Screenshot after a tick at a fixed animation-free state
      const shot = await cdp.send("Page.captureScreenshot", { format: "png" });
      mkdirSync("screenshots", { recursive: true });
      writeFileSync(`screenshots/login-${vp.name}.png`, Buffer.from(shot.data, "base64"));
    }

    console.log(JSON.stringify(metrics, null, 1));
  } finally {
    chrome.kill();
  }
}

main().catch((e) => { console.error("FATAL", e); process.exit(1); });
