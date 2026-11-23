/* SnipGIF — 100% client-side animated background removal.
 *
 * Pipeline: extract frames (GIF via gifuct-js, animated WebP via the
 * ImageDecoder API with a libwebp-WASM fallback, MP4 via <video>
 * seek-stepping) -> run an ONNX segmentation
 * model per frame with onnxruntime-web (WASM) -> composite the predicted mask
 * as the alpha channel -> preview + export (PNG sequence .zip, alpha WebM).
 *
 * Models: u2net and isnet-general-use — the same open-source models the rembg
 * project uses, with rembg's exact preprocessing (see preprocess()):
 *   - u2net: 320x320 input, mean [.485,.456,.406], std [.229,.224,.225]
 *   - isnet: 1024x1024 input, mean [.5,.5,.5], std [1,1,1]
 *   both: pixels scaled by the frame's own max, mask min-max normalized.
 * Model files: on the live Blob site each model is self-hosted, split
 * into ~20 MB chunks under assets/models/ (Cloudflare Pages caps single
 * files at 25 MB), fetched part by part at runtime and reassembled in
 * the browser. This standalone repo does not ship the chunks — the
 * loader tries ./assets/models/ first and automatically falls back to
 * the Hugging Face mirrors when the chunks are not there. The files are byte-identical to the official rembg GitHub
 * release (verified 2026-11-22: sizes 175,997,641 / 178,648,008 bytes and
 * MD5s 60024c5c889badc19c04ad937298a77b / fc16ebd8b0c10d971d3513d564d01e29
 * match rembg's published checksums; chunk reassembly re-verified by MD5
 * before deploy). Hugging Face mirrors are kept only as a last-resort
 * fallback if the same-origin chunks cannot be fetched.
 *
 * MODE / AUTO heuristic (kept deliberately simple): unless the user forces a
 * model, look at the middle frame — if its mean luminance is low (< 80/255)
 * the scene counts as "dark" and ISNet is used; otherwise u2net (best at
 * people holding props). The chosen model is always reported in the UI.
 *
 * Limits (stated in the UI): ~15 seconds of animation, max 120 frames,
 * frames wider than 1280px are downscaled, and very long animations are
 * downscaled further to keep browser memory sane.
 */

"use strict";

var ORT_VERSION = "1.20.1";
var ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@" + ORT_VERSION + "/dist/";

var MODELS = {
  u2net: {
    label: "PERSON + PROPS (u2net)",
    size: 320,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    clip: true,
    bytes: 175997641,
    parts: 9,
    chunkBase: "./assets/models/u2net.part",
    urls: [
      "https://huggingface.co/tomjackson2023/rembg/resolve/main/u2net.onnx",
      "https://huggingface.co/lucasfaesa/rembg-models/resolve/main/u2net.onnx"
    ]
  },
  isnet: {
    label: "DARK SCENE (ISNet)",
    size: 1024,
    mean: [0.5, 0.5, 0.5],
    std: [1.0, 1.0, 1.0],
    clip: false,
    bytes: 178648008,
    parts: 9,
    chunkBase: "./assets/models/isnet.part",
    urls: [
      "https://huggingface.co/tomjackson2023/rembg/resolve/main/isnet-general-use.onnx",
      "https://huggingface.co/lucasfaesa/rembg-models/resolve/main/isnet-general-use.onnx"
    ]
  }
};

var MAX_DURATION_MS = 15500;   // ~15 s cap, small grace included
var MAX_FRAMES = 120;
var MAX_WIDTH = 1280;
var MEMORY_BUDGET = 350 * 1024 * 1024; // bytes, both frame sets (before+after)

/* ---------------- DOM ---------------- */
function $(id) { return document.getElementById(id); }
var dropzone = $("dropzone"), fileInput = $("file-input"),
    fileInfo = $("file-info"), errorBox = $("error-box"),
    modeSelect = $("mode-select"), previewSelect = $("preview-select"),
    processBtn = $("process-btn"),
    progressWrap = $("progress-wrap"), progressText = $("progress-text"),
    progressBar = $("progress-bar"), modelStatus = $("model-status"),
    previewWrap = $("preview-wrap"), beforeCanvas = $("before-canvas"),
    afterCanvas = $("after-canvas"), afterBox = $("after-box"),
    playbackRow = $("playback-row"), playpauseBtn = $("playpause-btn"),
    frameReadout = $("frame-readout"),
    downloadRow = $("download-row"), downloadPngBtn = $("download-png"),
    downloadWebmBtn = $("download-webm"), webmNote = $("webm-note"),
    downloadStatus = $("download-status"),
    chipPng = $("chip-png"), chipWebm = $("chip-webm");

/* ---------------- State ---------------- */
var state = {
  fileName: "snipgif",
  frames: [],          // [{ imageData, delayMs }] source frames (composited)
  width: 0, height: 0,
  outputs: [],         // [{ imageData, delayMs }] cutout frames
  sessions: {},        // modelKey -> ort.InferenceSession
  modelBuffers: {},    // modelKey -> ArrayBuffer
  busy: false,
  playing: false,
  playTimer: null,
  playIndex: 0
};

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = !msg;
}
function setProgress(frac, text) {
  progressBar.style.width = Math.round(frac * 100) + "%";
  if (text) progressText.textContent = text;
}
function yieldToUI() { return new Promise(function (r) { setTimeout(r, 0); }); }

/* ---------------- File type sniffing ---------------- */
function sniffType(buf, name) {
  var b = new Uint8Array(buf.slice(0, 12));
  if (b[0] === 0x47 && b[1] === 0x49 && b[2] === 0x46) return "gif"; // GIF8
  if (b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46 &&
      b[8] === 0x57 && b[9] === 0x45 && b[10] === 0x42 && b[11] === 0x50) return "webp";
  var s = "";
  for (var i = 4; i < 8; i++) s += String.fromCharCode(b[i]);
  if (s === "ftyp") return "mp4";
  var ext = (name.split(".").pop() || "").toLowerCase();
  if (ext === "gif" || ext === "webp" || ext === "mp4") return ext;
  return null;
}

/* ---------------- Frame extraction ---------------- */
// A frame is { imageData, delayMs } — full-frame, already composited.

async function extractGif(buf) {
  var mod = await import("https://cdn.jsdelivr.net/npm/gifuct-js@2.1.2/+esm");
  var gif = mod.parseGIF(buf);
  var raw = mod.decompressFrames(gif, true);
  if (!raw.length) throw new Error("No frames found in this GIF.");
  // Canvas size = union of all frame rectangles (GIF patches can be partial).
  var w = 0, h = 0;
  raw.forEach(function (f) {
    w = Math.max(w, f.dims.left + f.dims.width);
    h = Math.max(h, f.dims.top + f.dims.height);
  });

  var canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  var ctx = canvas.getContext("2d", { willReadFrequently: true });
  var patch = document.createElement("canvas");
  var pctx = patch.getContext("2d", { willReadFrequently: true });
  var frames = [];
  var snapshot = null;

  for (var i = 0; i < raw.length; i++) {
    var f = raw[i];
    if (f.disposalType === 3) {
      snapshot = ctx.getImageData(0, 0, w, h);
    }
    patch.width = f.dims.width; patch.height = f.dims.height;
    var pd = pctx.createImageData(f.dims.width, f.dims.height);
    pd.data.set(f.patch);
    pctx.putImageData(pd, 0, 0);
    ctx.drawImage(patch, f.dims.left, f.dims.top);
    frames.push({
      imageData: ctx.getImageData(0, 0, w, h),
      delayMs: Math.max(20, f.delay || 100)
    });
    if (f.disposalType === 2) {
      ctx.clearRect(f.dims.left, f.dims.top, f.dims.width, f.dims.height);
    } else if (f.disposalType === 3 && snapshot) {
      ctx.putImageData(snapshot, 0, 0);
    }
  }
  return frames;
}

async function extractWebp(file) {
  var buf = await file.arrayBuffer();
  var primaryError = null;
  if ("ImageDecoder" in window) {
    try {
      var frames = await extractWebpViaImageDecoder(buf);
      if (frames.length) return frames;
      primaryError = new Error("ImageDecoder returned no frames.");
    } catch (e) { primaryError = e; }
  } else {
    primaryError = new Error("This browser has no ImageDecoder API.");
  }
  // Fallback: libwebp itself, compiled to WASM (libwebp-wasm, pinned below).
  // It demuxes and composites animated WebP on its own — no ImageDecoder
  // involved. This covers browsers without WebCodecs image decoding
  // (Firefox/Safari) AND files that strict demuxers refuse to expose a
  // track for while libwebp accepts them (the short36 test file is one:
  // Chrome's ImageDecoder reports no image track for it and ffmpeg rejects
  // it outright, yet libwebp/PIL decode all 37 frames — verified bit-exact
  // against PIL in Node, see build log 2026-11-22).
  try {
    return await extractWebpViaLibwebp(buf);
  } catch (fallbackError) {
    throw new Error("Could not decode this WebP file. ImageDecoder said: " +
      (primaryError && primaryError.message) + " The built-in libwebp decoder said: " +
      ((fallbackError && fallbackError.message) || fallbackError));
  }
}

var LIBWEBP_WASM_URL = "https://cdn.jsdelivr.net/npm/libwebp-wasm@0.1.6/+esm";

async function extractWebpViaLibwebp(buf) {
  var mod = await import(LIBWEBP_WASM_URL);
  var decoded = await mod.decode(new Uint8Array(buf));
  if (!decoded || !decoded.frameList || !decoded.frameList.length) {
    throw new Error("No frames found in this WebP.");
  }
  var frames = [];
  var prevTs = 0;
  for (var i = 0; i < decoded.frameList.length; i++) {
    var item = decoded.frameList[i];
    // libwebp reports each frame's cumulative END timestamp (ms since the
    // animation start); a frame's duration is the delta from the previous
    // frame's timestamp. Verified against the ANMF headers of the test
    // file: deltas are exactly its 66/67 ms frame durations.
    var delayMs = Math.max(20, Math.round(item.timestamp - prevTs) || 100);
    prevTs = item.timestamp;
    frames.push({
      imageData: new ImageData(item.data, decoded.width, decoded.height),
      delayMs: delayMs
    });
  }
  return frames;
}

async function extractWebpViaImageDecoder(buf) {
  var decoder = new ImageDecoder({ data: buf, type: "image/webp", preferAnimation: true });
  try {
    await decoder.completed;
    if (decoder.tracks && decoder.tracks.ready) {
      try { await decoder.tracks.ready; } catch (e) { /* track list stays best-effort */ }
    }
    var track = decoder.tracks.selectedTrack;
    if (!track) {
      // Chrome does not always pre-select a track for animated WebP, even
      // after `completed` resolves. Scan the track list for the first
      // usable image track and select it explicitly before giving up.
      for (var i = 0; i < decoder.tracks.length; i++) {
        var candidate = decoder.tracks[i];
        if (candidate && candidate.frameCount > 0) {
          candidate.selected = true;
          track = candidate;
          break;
        }
      }
      if (!track && decoder.tracks.length) {
        decoder.tracks[0].selected = true;
        track = decoder.tracks[0];
      }
    }
    if (!track) throw new Error("No image track found in this WebP file.");
    var count = track.frameCount;
    if (!count) throw new Error("The image track in this WebP reports no frames.");
    var frames = [];
    for (var i = 0; i < count; i++) {
      var result = await decoder.decode({ frameIndex: i });
      var img = result.image;
      var canvas = document.createElement("canvas");
      canvas.width = img.displayWidth; canvas.height = img.displayHeight;
      var ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0);
      var delayMs = img.duration ? Math.max(20, Math.round(img.duration / 1000)) : 100;
      frames.push({ imageData: ctx.getImageData(0, 0, canvas.width, canvas.height), delayMs: delayMs });
      img.close();
    }
    if (!frames.length) throw new Error("No frames found in this WebP.");
    return frames;
  } finally {
    decoder.close();
  }
}

function extractMp4(file) {
  return new Promise(function (resolve, reject) {
    var url = URL.createObjectURL(file);
    var video = document.createElement("video");
    video.muted = true; video.playsInline = true; video.preload = "auto";
    video.src = url;
    var done = false;
    function fail(msg) { if (!done) { done = true; URL.revokeObjectURL(url); reject(new Error(msg)); } }
    video.addEventListener("error", function () { fail("Could not read this MP4 file."); });
    // A damaged or half-downloaded file can leave the video element silent
    // forever — neither metadata nor an error event. Do not hang the UI.
    var metaTimer = setTimeout(function () { fail("Timed out waiting for this MP4's metadata — the file may be damaged."); }, 15000);
    video.addEventListener("loadedmetadata", async function () {
      clearTimeout(metaTimer);
      if (done) return;
      try {
        var duration = video.duration;
        if (!isFinite(duration) || duration <= 0) { fail("Could not read this MP4's duration."); return; }
        if (duration * 1000 > MAX_DURATION_MS) {
          fail("This clip is " + duration.toFixed(1) + "s — SnipGIF caps input at ~15 seconds. Trim it and try again.");
          return;
        }
        var count = Math.min(MAX_FRAMES, Math.max(1, Math.round(duration * 15)));
        var step = duration / count;
        var canvas = document.createElement("canvas");
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        var ctx = canvas.getContext("2d", { willReadFrequently: true });
        var frames = [];
        for (var i = 0; i < count; i++) {
          await seekTo(video, Math.min(i * step, duration - 0.01));
          ctx.drawImage(video, 0, 0);
          frames.push({
            imageData: ctx.getImageData(0, 0, canvas.width, canvas.height),
            delayMs: Math.round(step * 1000)
          });
        }
        done = true;
        URL.revokeObjectURL(url);
        resolve(frames);
      } catch (e) { fail(e.message || "MP4 decoding failed."); }
    });
  });
}
function seekTo(video, t) {
  return new Promise(function (resolve, reject) {
    // Already parked on the target time with data decoded (notably t=0 on
    // the first frame): some browsers fire no 'seeked' event for a no-op
    // seek, which would otherwise stall until the timeout below.
    if (video.readyState >= 2 && Math.abs(video.currentTime - t) < 0.002) { resolve(); return; }
    var timer = setTimeout(function () { reject(new Error("Timed out while reading the MP4.")); }, 8000);
    function onSeeked() {
      clearTimeout(timer);
      video.removeEventListener("seeked", onSeeked);
      resolve();
    }
    video.addEventListener("seeked", onSeeked);
    video.currentTime = t;
  });
}

/* Downsample to the frame cap, folding skipped frames' delays into the
   previous kept frame so total duration is preserved. */
function capFrames(frames) {
  if (frames.length <= MAX_FRAMES) return frames;
  var out = [];
  var step = frames.length / MAX_FRAMES;
  for (var i = 0; i < MAX_FRAMES; i++) {
    var start = Math.floor(i * step);
    var end = Math.floor((i + 1) * step);
    var delay = 0;
    for (var j = start; j < end; j++) delay += frames[j].delayMs;
    var f = frames[start];
    out.push({ imageData: f.imageData, delayMs: delay });
  }
  return out;
}

function scaleFrames(frames, scale) {
  if (scale >= 1) return frames;
  return frames.map(function (f) {
    var w = Math.max(1, Math.round(f.imageData.width * scale));
    var h = Math.max(1, Math.round(f.imageData.height * scale));
    var src = document.createElement("canvas");
    src.width = f.imageData.width; src.height = f.imageData.height;
    src.getContext("2d").putImageData(f.imageData, 0, 0);
    var dst = document.createElement("canvas");
    dst.width = w; dst.height = h;
    var dctx = dst.getContext("2d", { willReadFrequently: true });
    dctx.imageSmoothingEnabled = true; dctx.imageSmoothingQuality = "high";
    dctx.drawImage(src, 0, 0, w, h);
    return { imageData: dctx.getImageData(0, 0, w, h), delayMs: f.delayMs };
  });
}

async function loadFile(file) {
  stopPlayback();
  showError("");
  state.outputs = [];
  previewWrap.hidden = true; downloadRow.hidden = true; playbackRow.hidden = true;
  progressWrap.hidden = true;
  processBtn.disabled = true;
  fileInfo.hidden = true;

  var buf = await file.arrayBuffer();
  var type = sniffType(buf, file.name);
  if (!type) {
    showError("That file is not a GIF, animated WebP or MP4. SnipGIF only takes those three.");
    return;
  }
  var frames;
  try {
    fileInfo.hidden = false;
    fileInfo.textContent = "Reading " + file.name + " ...";
    if (type === "gif") frames = await extractGif(buf);
    else if (type === "webp") frames = await extractWebp(file);
    else frames = await extractMp4(file);
  } catch (e) {
    fileInfo.hidden = true;
    showError(e.message || "Could not read that file.");
    return;
  }

  var totalMs = frames.reduce(function (s, f) { return s + f.delayMs; }, 0);
  if (totalMs > MAX_DURATION_MS) {
    fileInfo.hidden = true;
    showError("That animation runs " + (totalMs / 1000).toFixed(1) + "s — SnipGIF caps input at ~15 seconds. Trim it and try again.");
    return;
  }
  frames = capFrames(frames);

  // Memory/size guards: max width, then a total-memory budget.
  var w = frames[0].imageData.width, h = frames[0].imageData.height;
  var scale = Math.min(1, MAX_WIDTH / w);
  var estBytes = w * scale * (h * scale) * frames.length * 8;
  if (estBytes > MEMORY_BUDGET) scale *= Math.sqrt(MEMORY_BUDGET / estBytes);
  if (scale < 1) frames = scaleFrames(frames, scale);

  state.frames = frames;
  state.width = frames[0].imageData.width;
  state.height = frames[0].imageData.height;
  state.fileName = (file.name.replace(/\.[^.]+$/, "") || "snipgif").replace(/[^a-z0-9-_]+/gi, "-");
  fileInfo.textContent = file.name + " — " + frames.length + " frames, " +
    state.width + "×" + state.height + ", " + (totalMs / 1000).toFixed(1) + "s. Pick a mode and hit REMOVE BACKGROUND.";
  processBtn.disabled = false;
}

/* ---------------- Model loading ---------------- */
/* Primary path: the model is self-hosted on this origin, split into
 * ~20 MB chunks (chunkBase + "00", "01", ...). Fetch the parts in order,
 * stream them into one buffer, and assert the reassembled length matches
 * the known model size before the buffer is trusted. */
async function fetchModelChunks(key) {
  var model = MODELS[key];
  var total = model.bytes;
  var buf = new Uint8Array(total);
  var received = 0;
  for (var p = 0; p < model.parts; p++) {
    var resp = await fetch(model.chunkBase + (p < 10 ? "0" : "") + p);
    if (!resp.ok) throw new Error("HTTP " + resp.status + " on model part " + (p + 1) + " of " + model.parts);
    var reader = resp.body.getReader();
    while (true) {
      var part = await reader.read();
      if (part.done) break;
      if (received + part.value.length > total) throw new Error("model parts are larger than the expected " + total + " bytes");
      buf.set(part.value, received);
      received += part.value.length;
      modelStatus.textContent = "Downloading the " + key.toUpperCase() + " model (one-time, ~" +
        Math.round(total / 1048576) + " MB) — " + Math.round(received / total * 100) + "%" +
        " (part " + (p + 1) + " of " + model.parts + ")...";
      await yieldToUI();
    }
  }
  if (received !== total) throw new Error("model download incomplete (" + received + " of " + total + " bytes)");
  return buf.buffer;
}

/* Last-resort fallback: the Hugging Face mirrors (a third-party host, kept
 * only for the case where the same-origin chunks cannot be fetched). */
async function fetchModelFromUrls(key) {
  var urls = MODELS[key].urls;
  var lastErr = null;
  for (var u = 0; u < urls.length; u++) {
    try {
      var resp = await fetch(urls[u]);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      var total = Number(resp.headers.get("Content-Length")) || 0;
      var reader = resp.body.getReader();
      var chunks = [], received = 0;
      while (true) {
        var part = await reader.read();
        if (part.done) break;
        chunks.push(part.value);
        received += part.value.length;
        modelStatus.textContent = "Downloading the " + key.toUpperCase() + " model (one-time, ~" +
          Math.round((total || received) / 1048576) + " MB) — " +
          (total ? Math.round(received / total * 100) + "%" : (received / 1048576).toFixed(0) + " MB") + "...";
        await yieldToUI();
      }
      if (total && received !== total) throw new Error("mirror download incomplete (" + received + " of " + total + " bytes)");
      var buf = new Uint8Array(received);
      var off = 0;
      chunks.forEach(function (c) { buf.set(c, off); off += c.length; });
      return buf.buffer;
    } catch (e) { lastErr = e; }
  }
  throw lastErr || new Error("no model source available");
}

async function fetchModel(key) {
  if (state.modelBuffers[key]) return state.modelBuffers[key];
  var buf = null, lastErr = null;
  try { buf = await fetchModelChunks(key); } catch (e) { lastErr = e; }
  if (!buf) {
    try { buf = await fetchModelFromUrls(key); } catch (e) { lastErr = e; }
  }
  if (!buf) throw new Error("Could not download the AI model (" + (lastErr && lastErr.message) + "). Check your connection and try again.");
  state.modelBuffers[key] = buf;
  return buf;
}

async function getSession(key) {
  if (state.sessions[key]) return state.sessions[key];
  if (!window.ort) throw new Error("The ONNX runtime failed to load. Check your connection and reload the page.");
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = 1; // no cross-origin isolation here; single-thread WASM is the reliable path
  var buf = await fetchModel(key);
  modelStatus.textContent = "Loading the " + key.toUpperCase() + " model into the browser...";
  var session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  state.sessions[key] = session;
  return session;
}

/* ---------------- Inference (rembg-exact preprocessing) ---------------- */
function preprocess(imageData, model) {
  var S = model.size;
  var src = document.createElement("canvas");
  src.width = imageData.width; src.height = imageData.height;
  src.getContext("2d").putImageData(imageData, 0, 0);
  var small = document.createElement("canvas");
  small.width = S; small.height = S;
  var sctx = small.getContext("2d", { willReadFrequently: true });
  sctx.imageSmoothingEnabled = true; sctx.imageSmoothingQuality = "high";
  sctx.drawImage(src, 0, 0, S, S);
  var px = sctx.getImageData(0, 0, S, S).data;

  var max = 0;
  for (var i = 0; i < px.length; i++) if (px[i] > max) max = px[i];
  if (max === 0) max = 1;

  var input = new Float32Array(3 * S * S);
  for (var p = 0; p < S * S; p++) {
    for (var c = 0; c < 3; c++) {
      input[c * S * S + p] = (px[p * 4 + c] / max - model.mean[c]) / model.std[c];
    }
  }
  return input;
}

async function predictMask(session, imageData, model) {
  var S = model.size;
  var inputName = session.inputNames[0];
  var feeds = {};
  feeds[inputName] = new ort.Tensor("float32", preprocess(imageData, model), [1, 3, S, S]);
  var out = await session.run(feeds);
  var data = out[session.outputNames[0]].data;

  var min = Infinity, max = -Infinity;
  for (var i = 0; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  var range = (max - min) || 1;
  // Mask at model size, as a grayscale canvas…
  var maskCanvas = document.createElement("canvas");
  maskCanvas.width = S; maskCanvas.height = S;
  var mctx = maskCanvas.getContext("2d", { willReadFrequently: true });
  var mid = mctx.createImageData(S, S);
  for (var j = 0; j < data.length; j++) {
    var v = (data[j] - min) / range;
    if (model.clip) v = Math.min(1, Math.max(0, v));
    var g = Math.round(v * 255);
    mid.data[j * 4] = g; mid.data[j * 4 + 1] = g; mid.data[j * 4 + 2] = g; mid.data[j * 4 + 3] = 255;
  }
  mctx.putImageData(mid, 0, 0);
  // …then resized back to frame size (rembg uses LANCZOS; browser smoothing is the close equivalent).
  var w = imageData.width, h = imageData.height;
  var full = document.createElement("canvas");
  full.width = w; full.height = h;
  var fctx = full.getContext("2d", { willReadFrequently: true });
  fctx.imageSmoothingEnabled = true; fctx.imageSmoothingQuality = "high";
  fctx.drawImage(maskCanvas, 0, 0, w, h);
  var maskPx = fctx.getImageData(0, 0, w, h).data;

  var result = new ImageData(w, h);
  var srcPx = imageData.data;
  for (var k = 0; k < srcPx.length; k += 4) {
    result.data[k] = srcPx[k];
    result.data[k + 1] = srcPx[k + 1];
    result.data[k + 2] = srcPx[k + 2];
    result.data[k + 3] = maskPx[k]; // mask's red channel == gray value
  }
  return result;
}

/* AUTO heuristic: dark middle frame -> ISNet, otherwise u2net. */
function pickAutoModel() {
  var f = state.frames[Math.floor(state.frames.length / 2)];
  var px = f.imageData.data;
  var sum = 0, n = 0;
  for (var i = 0; i < px.length; i += 16) { // sample every 4th pixel
    sum += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
    n++;
  }
  var meanLum = n ? sum / n : 255;
  return meanLum < 80 ? "isnet" : "u2net";
}

/* ---------------- Processing ---------------- */
async function processAll() {
  if (!state.frames.length || state.busy) return;
  state.busy = true;
  processBtn.disabled = true;
  showError("");
  downloadRow.hidden = true;
  progressWrap.hidden = false;
  setProgress(0, "Processing... frame 0 / " + state.frames.length);

  try {
    var mode = modeSelect.value;
    var key = mode === "auto" ? pickAutoModel() : mode;
    var model = MODELS[key];
    var session = await getSession(key);
    modelStatus.textContent = "Mode: " + (mode === "auto" ? "AUTO → " : "") + model.label + ". Running on your device...";

    var outputs = [];
    for (var i = 0; i < state.frames.length; i++) {
      setProgress(i / state.frames.length, "Processing... frame " + i + " / " + state.frames.length);
      outputs.push({
        imageData: await predictMask(session, state.frames[i].imageData, model),
        delayMs: state.frames[i].delayMs
      });
      await yieldToUI();
    }
    setProgress(1, "Done — " + state.frames.length + " frames cut.");
    modelStatus.textContent = "Model used: " + model.label + ". Nothing was uploaded.";
    state.outputs = outputs;
    showResults();
  } catch (e) {
    showError(e.message || "Processing failed.");
    progressWrap.hidden = true;
  }
  state.busy = false;
  processBtn.disabled = false;
}

function showResults() {
  previewWrap.hidden = false;
  playbackRow.hidden = false;
  downloadRow.hidden = false;
  beforeCanvas.width = state.width; beforeCanvas.height = state.height;
  afterCanvas.width = state.width; afterCanvas.height = state.height;
  checkWebmSupport();
  startPlayback();
}

/* ---------------- Preview playback ---------------- */
function drawFrame(idx) {
  if (!state.frames.length) return;
  beforeCanvas.getContext("2d").putImageData(state.frames[idx].imageData, 0, 0);
  if (state.outputs.length) {
    afterCanvas.getContext("2d").putImageData(state.outputs[idx].imageData, 0, 0);
  }
  frameReadout.textContent = "Frame " + (idx + 1) + " / " + state.frames.length;
}
function startPlayback() {
  stopPlayback();
  state.playing = true;
  playpauseBtn.innerHTML = "&#10073;&#10073; PAUSE PREVIEW";
  state.playIndex = 0;
  var step = function () {
    if (!state.playing) return;
    drawFrame(state.playIndex);
    var delay = state.frames[state.playIndex].delayMs;
    state.playIndex = (state.playIndex + 1) % state.frames.length;
    state.playTimer = setTimeout(step, Math.max(20, delay));
  };
  step();
}
function stopPlayback() {
  state.playing = false;
  if (state.playTimer) clearTimeout(state.playTimer);
  state.playTimer = null;
}
playpauseBtn.addEventListener("click", function () {
  if (!state.frames.length) return;
  if (state.playing) {
    stopPlayback();
    playpauseBtn.innerHTML = "&#9654; PLAY PREVIEW";
  } else {
    startPlayback();
  }
});

/* ---------------- Downloads ---------------- */
function downloadBlob(blob, filename) {
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 4000);
}
function frameToPngBlob(imageData) {
  return new Promise(function (resolve) {
    var c = document.createElement("canvas");
    c.width = imageData.width; c.height = imageData.height;
    c.getContext("2d").putImageData(imageData, 0, 0);
    c.toBlob(resolve, "image/png");
  });
}

downloadPngBtn.addEventListener("click", async function () {
  if (!state.outputs.length || !window.JSZip) return;
  downloadPngBtn.disabled = true;
  downloadStatus.textContent = "Building the ZIP...";
  try {
    var zip = new JSZip();
    var pad = String(state.outputs.length).length;
    for (var i = 0; i < state.outputs.length; i++) {
      var blob = await frameToPngBlob(state.outputs[i].imageData);
      zip.file("frame_" + String(i).padStart(pad, "0") + ".png", blob);
      downloadStatus.textContent = "Building the ZIP... " + (i + 1) + " / " + state.outputs.length;
    }
    var out = await zip.generateAsync({ type: "blob", compression: "STORE" });
    downloadBlob(out, state.fileName + "-cutout-png.zip");
    downloadStatus.textContent = "ZIP downloaded — " + state.outputs.length + " transparent PNG frames.";
  } catch (e) {
    downloadStatus.textContent = "ZIP failed: " + (e.message || e);
  }
  downloadPngBtn.disabled = false;
});

/* WebM alpha: WebCodecs VP9 encode with alpha kept, muxed with webm-muxer.
   Best effort by design — where the browser can't do it, the button is
   disabled with an honest note and the PNG sequence remains the output. */
var webmSupported = false;
async function checkWebmSupport() {
  webmSupported = false;
  downloadWebmBtn.disabled = true;
  webmNote.hidden = true;
  if (!("VideoEncoder" in window) || !state.outputs.length) { webmNote.hidden = false; return; }
  try {
    var cfg = {
      codec: "vp09.00.10.08",
      width: state.width, height: state.height,
      alpha: "keep", bitrate: 2_000_000, framerate: 30
    };
    var support = await VideoEncoder.isConfigSupported(cfg);
    webmSupported = !!support.supported;
  } catch (e) { webmSupported = false; }
  downloadWebmBtn.disabled = !webmSupported;
  webmNote.hidden = webmSupported;
}

downloadWebmBtn.addEventListener("click", async function () {
  if (!state.outputs.length || !webmSupported) return;
  downloadWebmBtn.disabled = true;
  downloadStatus.textContent = "Encoding the WebM...";
  try {
    var muxerMod = await import("https://cdn.jsdelivr.net/npm/webm-muxer@5.1.2/build/webm-muxer.mjs");
    var muxer = new muxerMod.Muxer({
      target: new muxerMod.ArrayBufferTarget(),
      video: { codec: "V_VP9", width: state.width, height: state.height, alpha: true }
    });
    var encoder = new VideoEncoder({
      output: function (chunk, meta) { muxer.addVideoChunk(chunk, meta); },
      error: function (e) { throw e; }
    });
    encoder.configure({
      codec: "vp09.00.10.08",
      width: state.width, height: state.height,
      alpha: "keep", bitrate: 2_000_000, framerate: 30
    });
    var canvas = document.createElement("canvas");
    canvas.width = state.width; canvas.height = state.height;
    var ctx = canvas.getContext("2d");
    var timestamp = 0;
    for (var i = 0; i < state.outputs.length; i++) {
      ctx.putImageData(state.outputs[i].imageData, 0, 0);
      var durationUs = Math.max(1, Math.round(state.outputs[i].delayMs * 1000));
      var frame = new VideoFrame(canvas, { timestamp: timestamp, duration: durationUs, alpha: "keep" });
      encoder.encode(frame, { keyFrame: i === 0 });
      frame.close();
      timestamp += durationUs;
      downloadStatus.textContent = "Encoding the WebM... " + (i + 1) + " / " + state.outputs.length;
      if (encoder.encodeQueueSize > 4) {
        await new Promise(function (r) {
          var t = setInterval(function () { if (encoder.encodeQueueSize <= 4) { clearInterval(t); r(); } }, 30);
        });
      }
    }
    await encoder.flush();
    muxer.finalize();
    var blob = new Blob([muxer.target.buffer], { type: "video/webm" });
    downloadBlob(blob, state.fileName + "-cutout-alpha.webm");
    downloadStatus.textContent = "WebM downloaded — VP9 with a real alpha channel.";
  } catch (e) {
    downloadStatus.textContent = "WebM encoding failed in this browser (" + (e.message || e) + "). The PNG sequence above always works.";
  }
  downloadWebmBtn.disabled = false;
});

/* ---------------- UI wiring ---------------- */
dropzone.addEventListener("click", function () { fileInput.click(); });
dropzone.addEventListener("keydown", function (e) {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
["dragenter", "dragover"].forEach(function (ev) {
  dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.add("sg-dragging"); });
});
["dragleave", "drop"].forEach(function (ev) {
  dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.remove("sg-dragging"); });
});
dropzone.addEventListener("drop", function (e) {
  var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) loadFile(f);
});
fileInput.addEventListener("change", function () {
  if (fileInput.files && fileInput.files[0]) loadFile(fileInput.files[0]);
  fileInput.value = "";
});
processBtn.addEventListener("click", processAll);
previewSelect.addEventListener("change", function () {
  afterBox.classList.toggle("sg-checker", previewSelect.value === "checker");
  document.querySelector(".sg-label-after").textContent =
    previewSelect.value === "checker" ? "AFTER — TRANSPARENT (SHOWN ON CHECKER)" : "AFTER — TRANSPARENT (SHOWN ON BLACK)";
});
function setOutputChoice(webm) {
  chipPng.classList.toggle("sg-chip-on", !webm);
  chipWebm.classList.toggle("sg-chip-on", webm);
  chipPng.setAttribute("aria-checked", String(!webm));
  chipWebm.setAttribute("aria-checked", String(webm));
  downloadPngBtn.classList.toggle("sg-primary-choice", !webm);
  downloadWebmBtn.classList.toggle("sg-primary-choice", webm);
}
chipPng.addEventListener("click", function () { setOutputChoice(false); });
chipWebm.addEventListener("click", function () { setOutputChoice(true); });
setOutputChoice(false);
