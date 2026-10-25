// Smart Study Monitor - mobile web version
// Detects: eyes closed too long, face hidden/missing, phone in view.
// Runs fully in the browser with MediaPipe Tasks Vision (no Python needed).

import {
  FilesetResolver,
  FaceLandmarker,
  ObjectDetector,
} from "./vision_bundle.mjs";

const WASM_URL = new URL(".", import.meta.url).href.replace(/\/$/, ""); // same folder as app.js
const FACE_MODEL = "./face_landmarker.task";
const OBJECT_MODEL =
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite";

const PHONE_INTERVAL_MS = 400; // phone check is heavier, so run it less often
const PHONE_HOLD_MS = 600;     // phone must be visible this long to alert
const PHONE_LOSS_MS = 1200;    // phone considered gone after this long unseen

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const banner = $("banner");
const statusEl = $("status");
const startBtn = $("startBtn");
const testBtn = $("testBtn");

// ---------- Settings (saved on the phone) ----------
const DEFAULTS = { sleep: 2, cover: 3, blink: 0.5, phone: 0.5, vibrate: true, front: true };
const settings = loadSettings();

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("ssm-settings") || "{}");
    return { ...DEFAULTS, ...saved };
  } catch (_) {
    return { ...DEFAULTS };
  }
}
function saveSettings() {
  try { localStorage.setItem("ssm-settings", JSON.stringify(settings)); } catch (_) {}
}

function bindSlider(id, labelId, key, fmt) {
  const el = $(id), label = $(labelId);
  el.value = settings[key];
  label.textContent = fmt(settings[key]);
  el.addEventListener("input", () => {
    settings[key] = parseFloat(el.value);
    label.textContent = fmt(settings[key]);
    saveSettings();
  });
}
bindSlider("sSleep", "vSleep", "sleep", (v) => v + " s");
bindSlider("sCover", "vCover", "cover", (v) => v + " s");
bindSlider("sBlink", "vBlink", "blink", (v) => Number(v).toFixed(2));
bindSlider("sPhone", "vPhone", "phone", (v) => Math.round(v * 100) + "%");
$("sVibrate").checked = settings.vibrate;
$("sFront").checked = settings.front;
$("sVibrate").addEventListener("change", (e) => { settings.vibrate = e.target.checked; saveSettings(); });
$("sFront").addEventListener("change", (e) => { settings.front = e.target.checked; saveSettings(); });

// ---------- Audio ----------
const sounds = {
  cover: new Audio("./faudio.mp3"),
  sleep: new Audio("./alarm.mp3"),
  phone: new Audio("./paudio.mp3"),
};
Object.values(sounds).forEach((a) => { a.preload = "auto"; });

function stopAllSounds() {
  Object.values(sounds).forEach((a) => { try { a.pause(); a.currentTime = 0; } catch (_) {} });
}
async function unlockAudio() {
  // Browsers only allow sound after a tap, so briefly "play" each file silently.
  for (const a of Object.values(sounds)) {
    try {
      a.muted = true;
      await a.play();
      a.pause();
      a.currentTime = 0;
    } catch (_) { /* ignore */ }
    a.muted = false;
  }
}

// ---------- State ----------
const state = {
  running: false,
  starting: false,
  stream: null,
  faceLm: null,
  objDet: null,
  phoneLoading: false,
  rafId: 0,
  wakeLock: null,
  lastVideoTime: -1,
  lastPhoneRun: 0,
  lastTick: 0,
  lastUi: 0,
  startedAt: 0,
  faceVisible: false,
  blink: 0,
  closedSince: 0,
  noFaceSince: 0,
  phoneSince: 0,
  phoneLastSeen: 0,
  activeAlert: null,
  focusMs: 0,
  alerts: 0,
};

const MESSAGES = {
  cover: "🙈 DON'T COVER YOUR FACE!",
  sleep: "😴 WAKE UP & STUDY!",
  phone: "📵 PUT THE PHONE AWAY!",
};

function setStatus(text) { statusEl.textContent = text; }

// ---------- Model loading ----------
function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(label + " took too long (>" + Math.round(ms / 1000) + "s)")), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

let fileset = null;

async function loadFaceModel() {
  if (state.faceLm) return;
  setStatus("Loading face detector (1/2)…");
  if (!fileset) fileset = await withTimeout(FilesetResolver.forVisionTasks(WASM_URL), 30000, "Engine load");
  // CPU is slower but far more reliable on budget phones than GPU.
  state.faceLm = await withTimeout(
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: "CPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
    }),
    45000,
    "Face model"
  );
}

// Phone detection is optional: if it can't load (slow/no internet) face monitoring still works.
async function loadPhoneModel() {
  if (state.objDet || state.phoneLoading) return;
  state.phoneLoading = true;
  try {
    if (!fileset) fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    state.objDet = await withTimeout(
      ObjectDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: OBJECT_MODEL, delegate: "CPU" },
        runningMode: "VIDEO",
        scoreThreshold: 0.3,
        maxResults: 5,
        categoryAllowlist: ["cell phone"],
      }),
      120000,
      "Phone model"
    );
    if (state.running) setStatus("Monitoring… (phone detection ready)");
  } catch (e) {
    console.warn("Phone detector unavailable", e);
    if (state.running) setStatus("Monitoring face & eyes. Phone detection unavailable (needs internet): " + e.message);
  } finally {
    state.phoneLoading = false;
  }
}

// ---------- Camera ----------
async function openCamera() {
  if (!window.isSecureContext) {
    throw new Error("Camera needs a secure page (http://localhost or https). See the guide.");
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("This browser cannot access the camera. Open this page in Chrome.");
  }
  const constraints = {
    audio: false,
    video: {
      facingMode: settings.front ? "user" : "environment",
      width: { ideal: 480 },
      height: { ideal: 360 },
      frameRate: { ideal: 24, max: 30 },
    },
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await new Promise((resolve) => {
    if (video.readyState >= 1) return resolve();
    video.onloadedmetadata = () => resolve();
  });
  await video.play();
  canvas.width = video.videoWidth || 480;
  canvas.height = video.videoHeight || 360;
  // Mirror only for the front camera so it feels like a mirror.
  $("mirror").style.transform = settings.front ? "scaleX(-1)" : "none";
  return stream;
}

async function requestWakeLock() {
  try {
    if ("wakeLock" in navigator) {
      state.wakeLock = await navigator.wakeLock.request("screen");
    }
  } catch (_) { /* optional feature */ }
}
function releaseWakeLock() {
  try { state.wakeLock && state.wakeLock.release(); } catch (_) {}
  state.wakeLock = null;
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && state.running) requestWakeLock();
});

// ---------- Analysis ----------
function analyseFace(now) {
  const res = state.faceLm.detectForVideo(video, now);
  if (res && res.faceLandmarks && res.faceLandmarks.length > 0) {
    state.faceVisible = true;
    state.noFaceSince = 0;

    const cats = (res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories) || [];
    let left = 0, right = 0;
    for (const c of cats) {
      if (c.categoryName === "eyeBlinkLeft") left = c.score;
      else if (c.categoryName === "eyeBlinkRight") right = c.score;
    }
    state.blink = (left + right) / 2;

    if (state.blink > settings.blink) {
      if (!state.closedSince) state.closedSince = now;
    } else {
      state.closedSince = 0;
    }
  } else {
    state.faceVisible = false;
    state.closedSince = 0;
    if (!state.noFaceSince) state.noFaceSince = now;
  }
}

function analysePhone(now) {
  if (!state.objDet) return;
  const res = state.objDet.detectForVideo(video, now);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let found = false;
  const dets = (res && res.detections) || [];
  for (const d of dets) {
    const cat = d.categories && d.categories[0];
    if (!cat || cat.categoryName !== "cell phone" || cat.score < settings.phone) continue;
    found = true;
    const b = d.boundingBox;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#f0f";
    ctx.strokeRect(b.originX, b.originY, b.width, b.height);
  }
  if (found) {
    if (!state.phoneSince || now - state.phoneLastSeen > PHONE_LOSS_MS) state.phoneSince = now;
    state.phoneLastSeen = now;
  }
}

function evaluate(now, dt) {
  const phoneActive = state.phoneLastSeen && now - state.phoneLastSeen < PHONE_LOSS_MS;
  if (!phoneActive) state.phoneSince = 0;

  let alert = null;
  if (state.noFaceSince && now - state.noFaceSince >= settings.cover * 1000) alert = "cover";
  else if (state.closedSince && now - state.closedSince >= settings.sleep * 1000) alert = "sleep";
  else if (phoneActive && state.phoneSince && now - state.phoneSince >= PHONE_HOLD_MS) alert = "phone";

  if (alert !== state.activeAlert) {
    stopAllSounds();
    state.activeAlert = alert;
    if (alert) {
      state.alerts += 1;
      banner.textContent = MESSAGES[alert];
      banner.className = "banner " + alert;
      banner.style.display = "block";
      if (settings.vibrate && navigator.vibrate) navigator.vibrate([300, 150, 300]);
    } else {
      banner.style.display = "none";
    }
  }

  if (alert) {
    const a = sounds[alert];
    if (a.paused || a.ended) {
      a.currentTime = 0;
      a.play().catch(() => {});
    }
  } else if (state.faceVisible) {
    state.focusMs += dt;
  }
}

function fmtTime(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return String(m).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
}

function updateUi(now) {
  if (now - state.lastUi < 250) return;
  state.lastUi = now;

  const face = $("chipFace"), eyes = $("chipEyes"), ph = $("chipPhone");
  face.textContent = state.faceVisible ? "Visible" : "Not found";
  face.className = state.faceVisible ? "ok" : "bad";

  if (state.faceVisible) {
    const closed = state.blink > settings.blink;
    eyes.textContent = (closed ? "Closed " : "Open ") + state.blink.toFixed(2);
    eyes.className = closed ? "warn" : "ok";
  } else {
    eyes.textContent = "–";
    eyes.className = "";
  }

  const phoneActive = state.phoneLastSeen && now - state.phoneLastSeen < PHONE_LOSS_MS;
  ph.textContent = phoneActive ? "Seen" : "None";
  ph.className = phoneActive ? "bad" : "ok";

  $("statTime").textContent = fmtTime(now - state.startedAt);
  $("statFocus").textContent = fmtTime(state.focusMs);
  $("statAlerts").textContent = String(state.alerts);
}

function loop() {
  if (!state.running) return;
  const now = performance.now();
  const dt = Math.min(now - state.lastTick, 500);
  state.lastTick = now;

  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    try { analyseFace(now); } catch (e) { console.error("face error", e); }
    if (now - state.lastPhoneRun >= PHONE_INTERVAL_MS) {
      state.lastPhoneRun = now;
      try { analysePhone(now); } catch (e) { console.error("phone error", e); }
    }
  }
  evaluate(now, dt);
  updateUi(now);
  state.rafId = requestAnimationFrame(loop);
}

// ---------- Start / Stop ----------
async function start() {
  if (state.running || state.starting) return;
  state.starting = true;
  startBtn.disabled = true;
  try {
    await unlockAudio();
    await loadFaceModel();
    setStatus("Opening camera…");
    state.stream = await openCamera();
    $("placeholder").style.display = "none";

    Object.assign(state, {
      lastVideoTime: -1, lastPhoneRun: 0, closedSince: 0, noFaceSince: 0,
      phoneSince: 0, phoneLastSeen: 0, activeAlert: null, focusMs: 0, alerts: 0,
      faceVisible: false, blink: 0,
    });
    state.startedAt = state.lastTick = performance.now();
    state.running = true;

    await requestWakeLock();
    startBtn.textContent = "■ Stop";
    startBtn.classList.add("stop");
    setStatus("Monitoring face & eyes… loading phone detector in background.");
    loop();
    loadPhoneModel();
  } catch (err) {
    console.error(err);
    setStatus("⚠️ " + friendlyError(err));
    cleanupCamera();
  } finally {
    state.starting = false;
    startBtn.disabled = false;
  }
}

function cleanupCamera() {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
}

function stop() {
  state.running = false;
  cancelAnimationFrame(state.rafId);
  cleanupCamera();
  stopAllSounds();
  releaseWakeLock();
  state.activeAlert = null;
  banner.style.display = "none";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("placeholder").style.display = "flex";
  startBtn.textContent = "▶ Start";
  startBtn.classList.remove("stop");
  setStatus("Stopped. Session: " + fmtTime(performance.now() - state.startedAt) +
    ", focused " + fmtTime(state.focusMs) + ", alerts " + state.alerts + ".");
}

function friendlyError(err) {
  const name = err && err.name;
  if (name === "NotAllowedError") return "Camera permission denied. Allow camera for this site in Chrome settings, then tap Start.";
  if (name === "NotFoundError") return "No camera found on this device.";
  if (name === "NotReadableError") return "Camera is busy. Close other apps using the camera and try again.";
  if (name === "OverconstrainedError") return "That camera isn't available. Try toggling 'Use front camera'.";
  const msg = (err && err.message) || String(err);
  if (/fetch|network|load|wasm|Failed/i.test(msg)) {
    return "Could not load AI models. Check your internet connection (needed the first time) and try again. (" + msg + ")";
  }
  return msg;
}

startBtn.addEventListener("click", () => (state.running ? stop() : start()));

testBtn.addEventListener("click", async () => {
  stopAllSounds();
  try {
    sounds.sleep.currentTime = 0;
    await sounds.sleep.play();
    setTimeout(stopAllSounds, 2500);
    setStatus("Playing test sound for 2 seconds…");
  } catch (e) {
    setStatus("⚠️ Couldn't play sound. Check media volume and that alarm.mp3 is in the same folder.");
  }
});

window.addEventListener("pagehide", () => { if (state.running) stop(); });
