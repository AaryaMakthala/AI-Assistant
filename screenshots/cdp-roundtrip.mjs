// Round-trip proof: drive the REAL app through the demo login, type a message in
// the composer, submit it, and confirm the user bubble AND an assistant response
// actually appear in the thread. Also samples the New chat button styles.
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const DEBUG_PORT = 9340;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function launchChrome() {
  return spawn(CHROME, [
    `--remote-debugging-port=${DEBUG_PORT}`,
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--user-data-dir=C:/Temp/cdp-rt-profile",
    "about:blank",
  ], { stdio: "ignore" });
}

async function getPageTarget() {
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
      const targets = await res.json();
      const page = targets.find((t) => t.type === "page");
      if (page) return page;
    } catch {}
    await sleep(250);
  }
  throw new Error("no chrome page target");
}

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id) {
        const p = this.pending.get(msg.id);
        if (p) {
          this.pending.delete(msg.id);
          msg.error ? p.reject(new Error(JSON.stringify(msg.error))) : p.resolve(msg.result);
        }
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
    await cdp.send("Runtime.enable");
    await cdp.send("Network.enable");
    await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });

    // Watch every backend call the app makes.
    const apiTraffic = [];
    const reqUrls = new Map();
    const handler = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id) return;
      if (m.method === "Network.requestWillBeSent") {
        reqUrls.set(m.params.requestId, m.params.request.url);
      } else if (m.method === "Network.responseReceived") {
        const url = reqUrls.get(m.params.requestId) || "";
        if (url.includes(":8000")) apiTraffic.push({ url: url.split("?")[0].replace(/^https?:\/\/[^/]+/, "").slice(0, 90), status: m.params.response.status });
      } else if (m.method === "Network.loadingFailed") {
        const url = reqUrls.get(m.params.requestId) || "";
        if (url.includes(":8000")) apiTraffic.push({ url: url.split("?")[0].replace(/^https?:\/\/[^/]+/, "").slice(0, 90), status: "LOAD_FAILED" });
      }
    };
    cdp.ws.addEventListener("message", handler);

    const evaljs = async (expression) => {
      const r = await cdp.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
      if (r.exceptionDetails) throw new Error("eval: " + JSON.stringify(r.exceptionDetails));
      return r.result.value;
    };

    // 1) Login page → click "Try the demo"
    await cdp.send("Page.navigate", { url: "http://localhost:3000/login" });
    await sleep(5000);
    const demoClicked = await evaljs(`(() => {
      const b = [...document.querySelectorAll("button")].find((x) => x.textContent.includes("Try the demo"));
      if (!b) return false;
      b.click();
      return true;
    })()`);
    console.log("demo button clicked:", demoClicked);

    // 2) Wait for redirect to the workspace and an enabled composer
    let authed = false;
    for (let i = 0; i < 90; i++) {
      const state = await evaljs(`(() => ({
        url: location.pathname,
        hasComposer: !!document.querySelector("textarea"),
        composerDisabled: document.querySelector("textarea") ? document.querySelector("textarea").disabled : null,
        demoError: document.querySelector(".login-demo-error")?.textContent ?? null,
        body: document.body.textContent.slice(0, 120),
      }))()`);
      if (state.url === "/" && state.composerDisabled === false) { authed = true; break; }
      if (state.demoError) { console.log("DEMO ERROR:", state.demoError); break; }
      await sleep(700);
    }
    console.log("authenticated workspace reached:", authed);

    const postLogin = await evaljs(`(() => ({
      sidebarTabs: [...document.querySelectorAll("aside [role=tablist] button")].map((b) => b.textContent.trim()),
      newChat: (() => {
        const b = [...document.querySelectorAll("aside button")].find((x) => x.textContent.includes("New chat"));
        if (!b) return null;
        const s = getComputedStyle(b);
        return { bg: s.backgroundColor, color: s.color, border: s.borderTopColor, radius: s.borderRadius };
      })(),
      workspaceName: document.querySelector("main h1 span")?.textContent ?? null,
      docCount: [...document.querySelectorAll("aside li")].length,
    }))()`);
    console.log("workspace state:", JSON.stringify(postLogin));

    if (!authed) {
      const err = await evaljs(`document.body.textContent.slice(0, 400)`);
      console.log("page body:", JSON.stringify(err.slice(0, 300)));
      await cdp.send("Page.captureScreenshot", { format: "png" }).then((s) => {
        mkdirSync("screenshots", { recursive: true });
        writeFileSync("screenshots/chat-roundtrip-login-failed.png", Buffer.from(s.data, "base64"));
      });
      return;
    }

    // 3) Click the textarea, type a real message through the input pipeline
    await evaljs(`(() => {
      const t = document.querySelector("textarea");
      const r = t.getBoundingClientRect();
      window.__ta = { x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2) };
      t.focus();
      return true;
    })()`);
    const ta = await evaljs(`window.__ta`);
    const focused = await evaljs(`document.activeElement === document.querySelector("textarea")`);
    console.log("textarea focused after click:", focused);

    await cdp.send("Input.insertText", { text: "hi" });
    await sleep(400);
    const typed = await evaljs(`document.querySelector("textarea").value`);
    console.log("typed value in textarea:", JSON.stringify(typed));
    // capture focused composer shot
    await cdp.send("Page.captureScreenshot", { format: "png" }).then((s) => {
      writeFileSync("screenshots/chat-roundtrip-typed.png", Buffer.from(s.data, "base64"));
    });

    // 4) Submit with Enter (same handler as the send button)
    await cdp.send("Input.dispatchKeyEvent", { type: "keyDown", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 });
    await cdp.send("Input.dispatchKeyEvent", { type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 });
    console.log("submitted");

    // 5) Poll the thread through the whole send lifecycle; also capture any
    // error banner and whether recovery reset the view.
    let turn = null;
    for (let i = 0; i < 90; i++) {
      const state = await evaljs(`(() => {
        const texts = [...document.querySelectorAll("main .bg-accent")].map((el) => el.textContent).filter(Boolean);
        const userMsg = texts.find((t) => t.trim() === "hi");
        const hasCard = document.querySelectorAll(".glass-message").length > 0;
        const cardText = hasCard ? [...document.querySelectorAll(".glass-message")].map((c) => c.textContent.trim()).join(" ") : "";
        const streaming = !!document.querySelector(".caret-blink");
        const alert = [...document.querySelectorAll("main [role=alert], main .text-danger")].map((el) => el.textContent.trim()).filter(Boolean).join(" | ");
        const authNotice = !!document.querySelector("main [role=status]");
        const emptyAgain = !hasCard && !userMsg && !document.querySelector(".caret-blink") && (document.body.textContent.includes("Ask about anything") || !texts.length);
        return { userMsg: !!userMsg, hasCard, len: cardText.length, cardText: cardText.slice(0, 120), streaming, alert, emptyAgain };
      })()`);
      if (state.emptyAgain) { console.log("!! thread went empty at poll", i); turn = { ...state, emptied: true }; break; }
      const settled = state.userMsg && state.hasCard && !state.streaming && state.len > 30;
      if (settled) { turn = state; break; }
      if (i % 10 === 0) console.log("poll", i, JSON.stringify({ user: state.userMsg, card: state.hasCard, len: state.len, streaming: state.streaming, alert: state.alert.slice(0, 60) }));
      await sleep(1000);
    }
    console.log("round trip result:", JSON.stringify(turn));
    console.log("API traffic:", JSON.stringify(apiTraffic, null, 1));

    await cdp.send("Page.captureScreenshot", { format: "png" }).then((s) => {
      writeFileSync("screenshots/chat-roundtrip-reply.png", Buffer.from(s.data, "base64"));
    });
  } finally {
    chrome.kill();
  }
}

main().catch((e) => { console.error("FATAL", e); process.exit(1); });