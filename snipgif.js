/* SnipGIF — 100% client-side animated background removal.
 *
 * Pipeline: extract frames (GIF via gifuct-js, animated WebP via the
 * ImageDecoder API with a libwebp-WASM fallback, MP4 via <video>
 * seek-stepping) -> run an ONNX segmentation
 * model per frame with onnxruntime-web (WASM) -> composite the predicted mask
 * as the alpha channel -> preview + export (transparent GIF, PNG sequence
 * .zip, alpha WebM). Files can be queued as a batch (one model load for the
 * whole queue), an optional sticker effect (wobble / shake / bounce) is
 * baked into every export, and when the page is cross-origin isolated the
 * WASM runtime uses up to 4 threads instead of 1.
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
    chipPng = $("chip-png"), chipWebm = $("chip-webm"), chipGif = $("chip-gif"),
    queueWrap = $("queue-wrap"), queueRows = $("queue-rows"),
    queueSummary = $("queue-summary"),
    downloadGifBtn = $("download-gif"), downloadAllBtn = $("download-all"),
    fxIntensity = $("fx-intensity"), fxIntensityVal = $("fx-intensity-val");
var afterLabel = document.querySelector(".sg-label-after");
var fxChips = {
  none: $("fx-none"), wobble: $("fx-wobble"),
  shake: $("fx-shake"), bounce: $("fx-bounce")
};

/* ---------------- State ----------------
 * The queue holds one item per file:
 *   { id, name, baseName, status, error, progress,
 *     frames, firstFrame, delays, frameCount, width, height, totalMs,
 *     outputs, cumMs, totalOutMs, thumb }
 * status: reading -> queued -> processing -> done | error.
 * state.frames / state.outputs / state.width / state.height mirror the
 * ACTIVE item so the preview + download code has one obvious data source. */
var MAX_QUEUE = 5;
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
  playIndex: 0,
  queue: [],
  activeId: 0,
  itemSeq: 0,
  outputFormat: "gif", // 'gif' | 'png' | 'webm' — the highlighted OUTPUT chip
  effect: "none",      // 'none' | 'wobble' | 'shake' | 'bounce'
  intensity: 5,        // 1..10
  fxCache: null,       // { item, effect, intensity, result } for export frames
  threads: 1           // WASM threads actually in use (set in getSession)
};

function activeItem() {
  for (var i = 0; i < state.queue.length; i++) {
    if (state.queue[i].id === state.activeId) return state.queue[i];
  }
  return null;
}

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

/* ---------------- Batch queue ---------------- */
/* Read + validate one file into a queue item. Throws Error with a
 * user-facing message on any problem; the caller records it on the item
 * and keeps going with the rest of the batch. */
async function readItemFrames(item, file) {
  var buf = await file.arrayBuffer();
  var type = sniffType(buf, file.name);
  if (!type) throw new Error("Not a GIF, animated WebP or MP4.");
  var frames;
  if (type === "gif") frames = await extractGif(buf);
  else if (type === "webp") frames = await extractWebp(file);
  else frames = await extractMp4(file);

  var totalMs = frames.reduce(function (s, f) { return s + f.delayMs; }, 0);
  if (totalMs > MAX_DURATION_MS) {
    throw new Error("Runs " + (totalMs / 1000).toFixed(1) + "s — over the ~15s cap.");
  }
  frames = capFrames(frames);

  // Memory/size guards: max width, then a total-memory budget.
  var w = frames[0].imageData.width, h = frames[0].imageData.height;
  var scale = Math.min(1, MAX_WIDTH / w);
  var estBytes = w * scale * (h * scale) * frames.length * 8;
  if (estBytes > MEMORY_BUDGET) scale *= Math.sqrt(MEMORY_BUDGET / estBytes);
  if (scale < 1) frames = scaleFrames(frames, scale);

  item.frames = frames;
  item.firstFrame = frames[0];
  item.delays = frames.map(function (f) { return f.delayMs; });
  item.frameCount = frames.length;
  item.width = frames[0].imageData.width;
  item.height = frames[0].imageData.height;
  item.totalMs = totalMs;
  item.thumb = makeThumb(frames[0].imageData);
}

function makeThumb(imageData) {
  try {
    var src = document.createElement("canvas");
    src.width = imageData.width; src.height = imageData.height;
    src.getContext("2d").putImageData(imageData, 0, 0);
    var c = document.createElement("canvas");
    c.width = 44; c.height = 44;
    var ctx = c.getContext("2d");
    var s = Math.max(44 / imageData.width, 44 / imageData.height);
    var dw = imageData.width * s, dh = imageData.height * s;
    ctx.drawImage(src, (44 - dw) / 2, (44 - dh) / 2, dw, dh);
    return c.toDataURL("image/png");
  } catch (e) { return ""; }
}

function queueCounts() {
  var c = { done: 0, processing: 0, queued: 0, error: 0, reading: 0, total: state.queue.length };
  state.queue.forEach(function (it) {
    if (it.status === "done") c.done++;
    else if (it.status === "processing") c.processing++;
    else if (it.status === "queued") c.queued++;
    else if (it.status === "error") c.error++;
    else if (it.status === "reading") c.reading++;
  });
  return c;
}

function renderQueue() {
  var c = queueCounts();
  queueWrap.hidden = state.queue.length === 0;
  var parts = [];
  if (c.done) parts.push(c.done + " done");
  if (c.processing) parts.push(c.processing + " processing");
  if (c.queued) parts.push(c.queued + " queued");
  if (c.error) parts.push(c.error + " error");
  if (c.reading) parts.push(c.reading + " reading");
  queueSummary.textContent = parts.join(" · ");

  queueRows.innerHTML = "";
  state.queue.forEach(function (it) {
    var row = document.createElement("div");
    row.className = "sg-qrow" + (it.id === state.activeId ? " sg-qrow-active" : "");
    var meta = "";
    if (it.status === "reading") meta = "reading...";
    else if (it.status === "error") meta = it.error || "Could not read this file.";
    else meta = it.frameCount + " frames · " + it.width + "×" + it.height;
    var statusHtml = "", statusCls = "sg-qstatus";
    if (it.status === "done") { statusHtml = "&#10003; DONE"; statusCls += " sg-done"; }
    else if (it.status === "processing") { statusHtml = "PROCESSING " + (it.progress || 0) + "%"; statusCls += " sg-proc"; }
    else if (it.status === "error") { statusHtml = "ERROR"; statusCls += " sg-err"; }
    else if (it.status === "reading") statusHtml = "READING";
    else statusHtml = "QUEUED";
    row.innerHTML =
      (it.thumb ? '<img class="sg-qthumb" alt="" src="' + it.thumb + '">' : '<span class="sg-qthumb"></span>') +
      '<span class="sg-qmain"><span class="sg-qname"></span><br><span class="sg-qmeta"></span>' +
      (it.status === "processing" ? '<span class="sg-qbar"><span style="width:' + (it.progress || 0) + '%"></span></span>' : "") +
      '</span>' +
      '<span class="' + statusCls + '">' + statusHtml + '</span>' +
      '<button type="button" class="sg-qremove" title="Remove from queue" aria-label="Remove ' +
      it.name.replace(/"/g, "&quot;") + '">&times;</button>';
    row.querySelector(".sg-qname").textContent = it.name;
    row.querySelector(".sg-qmeta").textContent = meta;
    row.querySelector(".sg-qremove").addEventListener("click", function (e) {
      e.stopPropagation();
      removeItem(it.id);
    });
    row.addEventListener("click", function () {
      if (it.status === "done" || it.status === "queued") setActive(it, true);
    });
    queueRows.appendChild(row);
  });
  updateDownloadAll();
}

function updateDownloadAll() {
  var c = queueCounts();
  var show = c.done >= 2;
  downloadAllBtn.hidden = !show;
  if (show) downloadAllBtn.textContent = "DOWNLOAD ALL — " + c.done + " FILES (.ZIP)";
}

function updateProcessBtn() {
  var c = queueCounts();
  var pending = c.queued;
  processBtn.disabled = state.busy || pending === 0;
  processBtn.textContent = pending > 1 ? "REMOVE BACKGROUND — " + pending + " FILES" : "REMOVE BACKGROUND";
}

function updateFileInfo() {
  var c = queueCounts();
  if (!state.queue.length) { fileInfo.hidden = true; return; }
  fileInfo.hidden = false;
  if (state.queue.length === 1) {
    var it = state.queue[0];
    if (it.status === "queued" || it.status === "done") {
      fileInfo.textContent = it.name + " — " + it.frameCount + " frames, " + it.width + "×" + it.height +
        ", " + (it.totalMs / 1000).toFixed(1) + "s. Pick a mode and hit REMOVE BACKGROUND.";
      return;
    }
    if (it.status === "reading") { fileInfo.textContent = "Reading " + it.name + " ..."; return; }
    if (it.status === "error") { fileInfo.hidden = true; return; }
  }
  var bits = [];
  if (c.queued) bits.push(c.queued + " waiting");
  if (c.done) bits.push(c.done + " done");
  if (c.error) bits.push(c.error + " could not be read (see the queue)");
  fileInfo.textContent = state.queue.length + " files in the queue — " + bits.join(", ") +
    ". Hit REMOVE BACKGROUND to cut them all; the model loads once for the whole batch.";
}

/* Mirror an item into state.* and (when asked) refresh the preview area. */
function setActive(item, show) {
  if (!item) return;
  state.activeId = item.id;
  state.frames = item.frames || [];
  state.outputs = item.outputs || [];
  state.width = item.width || 0;
  state.height = item.height || 0;
  state.fileName = item.baseName;
  state.fxCache = null;
  if (show) {
    if (item.outputs && item.outputs.length) showResults();
    else {
      stopPlayback();
      previewWrap.hidden = true; downloadRow.hidden = true; playbackRow.hidden = true;
    }
  }
  renderQueue();
}

function removeItem(id) {
  if (state.busy) return;
  var idx = -1;
  state.queue.forEach(function (it, i) { if (it.id === id) idx = i; });
  if (idx < 0) return;
  var wasActive = state.activeId === id;
  state.queue.splice(idx, 1);
  if (wasActive) {
    state.activeId = 0;
    state.frames = []; state.outputs = []; state.fxCache = null;
    stopPlayback();
    previewWrap.hidden = true; downloadRow.hidden = true; playbackRow.hidden = true;
    var next = state.queue.filter(function (it) { return it.status === "done"; })[0] ||
               state.queue.filter(function (it) { return it.status === "queued"; })[0];
    if (next) setActive(next, true);
  }
  renderQueue();
  updateProcessBtn();
  updateFileInfo();
}

async function addFiles(fileList) {
  var files = Array.prototype.slice.call(fileList || []);
  if (!files.length) return;
  showError("");
  progressWrap.hidden = true;
  var skipped = 0;
  for (var i = 0; i < files.length; i++) {
    if (state.queue.length >= MAX_QUEUE) { skipped = files.length - i; break; }
    var file = files[i];
    var item = {
      id: ++state.itemSeq,
      name: file.name,
      baseName: (file.name.replace(/\.[^.]+$/, "") || "snipgif").replace(/[^a-z0-9-_]+/gi, "-"),
      status: "reading", error: "", progress: 0,
      frames: null, firstFrame: null, delays: [], frameCount: 0,
      width: 0, height: 0, totalMs: 0,
      outputs: null, cumMs: null, totalOutMs: 0, thumb: ""
    };
    state.queue.push(item);
    renderQueue();
    updateFileInfo();
    try {
      await readItemFrames(item, file);
      item.status = "queued";
    } catch (e) {
      item.status = "error";
      item.error = e.message || "Could not read that file.";
    }
    renderQueue();
    updateProcessBtn();
    updateFileInfo();
  }
  if (skipped > 0) {
    showError("The batch is capped at " + MAX_QUEUE + " files per round — " + skipped + " extra file(s) were skipped.");
  }
  if (!activeItem()) {
    var first = state.queue.filter(function (it) { return it.status === "done"; })[0] ||
                state.queue.filter(function (it) { return it.status === "queued"; })[0];
    if (first) setActive(first, false);
  }
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
  // Multi-threaded WASM needs cross-origin isolation (SharedArrayBuffer).
  // The live Blob site sends COOP/COEP headers for /snipgif/ only (via a
  // _headers file), so where isolation is active we fan out to up to 4 threads;
  // everywhere else this stays at 1 — the previous reliable behaviour.
  var threads = 1;
  if (self.crossOriginIsolated && navigator.hardwareConcurrency) {
    threads = Math.min(4, navigator.hardwareConcurrency);
  }
  ort.env.wasm.numThreads = threads;
  state.threads = threads;
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
function pickAutoModel(item) {
  var frames = (item && item.frames) || state.frames;
  var f = frames[Math.floor(frames.length / 2)];
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
function computeCum(item) {
  var cum = [], t = 0;
  (item.outputs || []).forEach(function (o) { cum.push(t); t += o.delayMs; });
  item.cumMs = cum;
  item.totalOutMs = t || 1;
}

/* Once a batch grows large, holding every file's SOURCE frames in RAM gets
 * reckless. When retained source bytes pass ~500 MB, drop the source frames
 * of finished, non-active items (their cutouts, first frame and delays are
 * kept, so preview of the AFTER side and all downloads still work; only the
 * BEFORE-side animation of those items becomes a still first frame). */
function maybeFreeFrames() {
  var total = 0;
  state.queue.forEach(function (it) {
    if (it.frames) total += it.width * it.height * 4 * it.frames.length;
  });
  if (total <= 500 * 1024 * 1024) return;
  state.queue.forEach(function (it) {
    if (total <= 500 * 1024 * 1024) return;
    if (it.status === "done" && it.id !== state.activeId && it.frames) {
      total -= it.width * it.height * 4 * it.frames.length;
      it.frames = null;
    }
  });
}

async function processAll() {
  var pending = state.queue.filter(function (it) { return it.status === "queued"; });
  if (!pending.length || state.busy) return;
  state.busy = true;
  updateProcessBtn();
  showError("");
  downloadRow.hidden = true;
  progressWrap.hidden = false;
  setProgress(0, "Processing...");

  try {
    var mode = modeSelect.value;
    var lastModelLabel = "";
    for (var qi = 0; qi < pending.length; qi++) {
      var item = pending[qi];
      item.status = "processing";
      item.progress = 0;
      setActive(item, false);
      renderQueue();
      var key = mode === "auto" ? pickAutoModel(item) : mode;
      var model = MODELS[key];
      lastModelLabel = model.label;
      var session = await getSession(key);
      modelStatus.textContent = "Mode: " + (mode === "auto" ? "AUTO → " : "") + model.label +
        ". Running on your device" + (state.threads > 1 ? " with " + state.threads + " threads" : "") + "...";

      var outputs = [];
      var n = item.frames.length;
      for (var i = 0; i < n; i++) {
        var label = (pending.length > 1 ? "Processing file " + (qi + 1) + " of " + pending.length + " — frame " : "Processing... frame ") +
          i + " / " + n + (state.threads > 1 ? " · " + state.threads + " threads" : "");
        setProgress((qi + i / n) / pending.length, label);
        item.progress = Math.round(i / n * 100);
        if (i % 4 === 0) renderQueue();
        outputs.push({
          imageData: await predictMask(session, item.frames[i].imageData, model),
          delayMs: item.frames[i].delayMs
        });
        await yieldToUI();
      }
      item.outputs = outputs;
      computeCum(item);
      item.status = "done";
      item.progress = 100;
      if (item.id === state.activeId) state.outputs = outputs;
      renderQueue();
      maybeFreeFrames();
    }
    setProgress(1, pending.length > 1 ? "Done — " + pending.length + " files cut." : "Done — " + pending[0].frameCount + " frames cut.");
    modelStatus.textContent = "Model used: " + lastModelLabel + ". Nothing was uploaded.";
    setActive(pending[pending.length - 1], true);
  } catch (e) {
    state.queue.forEach(function (it) { if (it.status === "processing") it.status = "queued"; });
    renderQueue();
    showError(e.message || "Processing failed.");
    progressWrap.hidden = true;
  }
  state.busy = false;
  updateProcessBtn();
}

function showResults() {
  var item = activeItem();
  if (!item || !item.outputs || !item.outputs.length) return;
  previewWrap.hidden = false;
  playbackRow.hidden = false;
  downloadRow.hidden = false;
  beforeCanvas.width = item.width; beforeCanvas.height = item.height;
  var dims = exportDimsFor(item);
  afterCanvas.width = dims.w; afterCanvas.height = dims.h;
  checkWebmSupport();
  updateDownloadAll();
  startPlayback();
}

/* ---------------- Sticker effects ----------------
 * One shared transform drives BOTH the preview and every export, so what
 * you see is exactly what downloads. The phase is derived from the frame's
 * position in the loop (cumulative delays), with an integer number of
 * cycles per loop, so exported GIFs loop seamlessly. SHAKE uses a
 * deterministic per-frame hash — never Math.random — for the same reason:
 * preview and export must agree frame by frame. */
var FX_PAD_RATIO = 0.09; // transparent margin added around the frame when an effect is on

function fxActive() { return state.effect !== "none"; }

function exportDimsFor(item) {
  var pad = fxActive() ? Math.ceil(Math.max(item.width, item.height) * FX_PAD_RATIO) : 0;
  return { w: item.width + 2 * pad, h: item.height + 2 * pad, pad: pad };
}

function fxHash(n) {
  var x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function fxTransform(item, idx) {
  var t = { angle: 0, dx: 0, dy: 0, sx: 1, sy: 1 };
  if (!fxActive() || !item.cumMs) return t;
  var k = state.intensity / 10;
  var p = item.cumMs[idx] / item.totalOutMs; // 0..1 across the loop
  if (state.effect === "wobble") {
    t.angle = Math.sin(p * Math.PI * 4) * (state.intensity * 1.1) * Math.PI / 180;
  } else if (state.effect === "shake") {
    var amp = k * 0.02 * Math.max(item.width, item.height);
    t.dx = (fxHash(idx * 2) - 0.5) * 2 * amp;
    t.dy = (fxHash(idx * 2 + 1) - 0.5) * 2 * amp;
  } else if (state.effect === "bounce") {
    var b = Math.abs(Math.sin(p * Math.PI * 4));
    var land = Math.pow(1 - b, 3);
    t.dy = -b * k * 0.06 * item.height;
    t.sy = 1 - 0.10 * k * land;
    t.sx = 1 + 0.06 * k * land;
  }
  return t;
}

var fxTmpCanvas = document.createElement("canvas");
function drawTransformed(ctx, imageData, item, idx, dims) {
  fxTmpCanvas.width = imageData.width; fxTmpCanvas.height = imageData.height;
  fxTmpCanvas.getContext("2d").putImageData(imageData, 0, 0);
  var t = fxTransform(item, idx);
  ctx.clearRect(0, 0, dims.w, dims.h);
  ctx.save();
  ctx.translate(dims.w / 2 + t.dx, dims.h / 2 + t.dy);
  ctx.rotate(t.angle);
  ctx.scale(t.sx, t.sy);
  ctx.drawImage(fxTmpCanvas, -imageData.width / 2, -imageData.height / 2);
  ctx.restore();
}

/* The item's cutout frames with the current effect baked in, at export
 * dimensions. Cached per (item, effect, intensity); the no-effect path
 * returns the raw outputs untouched. */
function getExportFrames(item) {
  if (!item || !item.outputs || !item.outputs.length) return null;
  if (!fxActive()) return { frames: item.outputs, width: item.width, height: item.height };
  var cache = state.fxCache;
  if (cache && cache.item === item && cache.effect === state.effect && cache.intensity === state.intensity) {
    return cache.result;
  }
  var dims = exportDimsFor(item);
  var c = document.createElement("canvas");
  c.width = dims.w; c.height = dims.h;
  var ctx = c.getContext("2d", { willReadFrequently: true });
  var frames = item.outputs.map(function (o, i) {
    drawTransformed(ctx, o.imageData, item, i, dims);
    return { imageData: ctx.getImageData(0, 0, dims.w, dims.h), delayMs: o.delayMs };
  });
  var result = { frames: frames, width: dims.w, height: dims.h };
  state.fxCache = { item: item, effect: state.effect, intensity: state.intensity, result: result };
  return result;
}

function fxSuffix() {
  if (!fxActive()) return "";
  return " · effect: " + state.effect.toUpperCase() + " · intensity " + state.intensity;
}

function onFxChanged() {
  state.fxCache = null;
  var item = activeItem();
  if (!item || !item.outputs || !item.outputs.length) return;
  var dims = exportDimsFor(item);
  afterCanvas.width = dims.w; afterCanvas.height = dims.h;
  drawFrame(state.playIndex || 0);
  checkWebmSupport();
}

/* ---------------- Preview playback ---------------- */
function drawAfterFrame(idx) {
  var item = activeItem();
  if (!item || !item.outputs || !item.outputs.length) return;
  var dims = exportDimsFor(item);
  if (afterCanvas.width !== dims.w || afterCanvas.height !== dims.h) {
    afterCanvas.width = dims.w; afterCanvas.height = dims.h;
  }
  var ctx = afterCanvas.getContext("2d");
  if (!fxActive()) {
    ctx.putImageData(item.outputs[idx].imageData, 0, 0);
    return;
  }
  drawTransformed(ctx, item.outputs[idx].imageData, item, idx, dims);
}

function drawFrame(idx) {
  var item = activeItem();
  if (!item || !item.frameCount) return;
  idx = Math.min(idx, item.frameCount - 1);
  var src = item.frames || (item.firstFrame ? [item.firstFrame] : []);
  if (src.length) {
    beforeCanvas.getContext("2d").putImageData(src[Math.min(idx, src.length - 1)].imageData, 0, 0);
  }
  if (item.outputs && item.outputs.length) drawAfterFrame(idx);
  frameReadout.textContent = "Frame " + (idx + 1) + " / " + item.frameCount + fxSuffix();
}
function startPlayback() {
  var item = activeItem();
  if (!item || !item.frameCount) return;
  stopPlayback();
  state.playing = true;
  playpauseBtn.innerHTML = "&#10073;&#10073; PAUSE PREVIEW";
  state.playIndex = 0;
  var step = function () {
    if (!state.playing) return;
    var cur = activeItem();
    if (!cur || !cur.frameCount) { stopPlayback(); return; }
    drawFrame(state.playIndex);
    var delay = cur.delays[state.playIndex] || 100;
    state.playIndex = (state.playIndex + 1) % cur.frameCount;
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
  var item = activeItem();
  if (!item || !item.frameCount) return;
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

async function addPngFramesToZip(zip, item) {
  var exp = getExportFrames(item);
  var pad = String(exp.frames.length).length;
  for (var i = 0; i < exp.frames.length; i++) {
    var blob = await frameToPngBlob(exp.frames[i].imageData);
    zip.file("frame_" + String(i).padStart(pad, "0") + ".png", blob);
  }
}

downloadPngBtn.addEventListener("click", async function () {
  var item = activeItem();
  if (!item || !item.outputs || !item.outputs.length || !window.JSZip) return;
  downloadPngBtn.disabled = true;
  downloadStatus.textContent = "Building the ZIP...";
  try {
    var zip = new JSZip();
    await addPngFramesToZip(zip, item);
    var out = await zip.generateAsync({ type: "blob", compression: "STORE" });
    downloadBlob(out, item.baseName + "-cutout-png.zip");
    downloadStatus.textContent = "ZIP downloaded — " + item.outputs.length + " transparent PNG frames.";
  } catch (e) {
    downloadStatus.textContent = "ZIP failed: " + (e.message || e);
  }
  downloadPngBtn.disabled = false;
});

/* Transparent GIF: gifenc (MIT) encodes in the browser. GIF alpha is
 * 1-bit, so the model's soft alpha is snapped at 128 first — pixels below
 * become fully transparent (RGB zeroed, so no colour fringe survives),
 * pixels above become fully opaque. Edges are therefore harder than in
 * the PNG/WebM exports; the UI says so next to the button. */
async function encodeGifBlob(item, statusPrefix) {
  var exp = getExportFrames(item);
  var mod = await import("https://cdn.jsdelivr.net/npm/gifenc@1.0.3/+esm");
  var gif = mod.GIFEncoder();
  for (var i = 0; i < exp.frames.length; i++) {
    var d = exp.frames[i].imageData.data;
    var rgba = new Uint8Array(d.length);
    for (var p = 0; p < d.length; p += 4) {
      if (d[p + 3] >= 128) {
        rgba[p] = d[p]; rgba[p + 1] = d[p + 1]; rgba[p + 2] = d[p + 2]; rgba[p + 3] = 255;
      }
    }
    var palette = mod.quantize(rgba, 256, { format: "rgba4444" });
    var index = mod.applyPalette(rgba, palette, "rgba4444");
    gif.writeFrame(index, exp.width, exp.height, {
      palette: palette,
      delay: exp.frames[i].delayMs,
      transparent: true,
      repeat: i === 0 ? 0 : undefined,
      first: i === 0
    });
    if (statusPrefix && i % 6 === 5) {
      downloadStatus.textContent = statusPrefix + " " + (i + 1) + " / " + exp.frames.length;
      await yieldToUI();
    }
  }
  gif.finish();
  return new Blob([gif.bytes()], { type: "image/gif" });
}

downloadGifBtn.addEventListener("click", async function () {
  var item = activeItem();
  if (!item || !item.outputs || !item.outputs.length) return;
  downloadGifBtn.disabled = true;
  downloadStatus.textContent = "Encoding the GIF...";
  try {
    var blob = await encodeGifBlob(item, "Encoding the GIF...");
    downloadBlob(blob, item.baseName + "-cutout.gif");
    downloadStatus.textContent = "GIF downloaded — transparent, " + item.outputs.length + " frames, " +
      (blob.size / 1048576).toFixed(1) + " MB.";
  } catch (e) {
    downloadStatus.textContent = "GIF encoding failed: " + (e.message || e);
  }
  downloadGifBtn.disabled = false;
});

/* WebM alpha: WebCodecs VP9 encode with alpha kept, muxed with webm-muxer.
   Best effort by design — where the browser can't do it, the button is
   disabled with an honest note and the PNG sequence remains the output. */
var webmSupported = false;
async function webmConfigOk(w, h) {
  if (!("VideoEncoder" in window)) return false;
  try {
    var support = await VideoEncoder.isConfigSupported({
      codec: "vp09.00.10.08",
      width: w, height: h,
      alpha: "keep", bitrate: 2_000_000, framerate: 30
    });
    return !!support.supported;
  } catch (e) { return false; }
}
async function checkWebmSupport() {
  webmSupported = false;
  downloadWebmBtn.disabled = true;
  webmNote.hidden = true;
  var item = activeItem();
  if (!item || !item.outputs || !item.outputs.length) { webmNote.hidden = false; return; }
  var dims = exportDimsFor(item);
  webmSupported = await webmConfigOk(dims.w, dims.h);
  downloadWebmBtn.disabled = !webmSupported;
  webmNote.hidden = webmSupported;
}

async function encodeWebmBlob(item, statusPrefix) {
  var exp = getExportFrames(item);
  var muxerMod = await import("https://cdn.jsdelivr.net/npm/webm-muxer@5.1.2/build/webm-muxer.mjs");
  var muxer = new muxerMod.Muxer({
    target: new muxerMod.ArrayBufferTarget(),
    video: { codec: "V_VP9", width: exp.width, height: exp.height, alpha: true }
  });
  var encoderError = null;
  var encoder = new VideoEncoder({
    output: function (chunk, meta) { muxer.addVideoChunk(chunk, meta); },
    error: function (e) { encoderError = e; }
  });
  encoder.configure({
    codec: "vp09.00.10.08",
    width: exp.width, height: exp.height,
    alpha: "keep", bitrate: 2_000_000, framerate: 30
  });
  var canvas = document.createElement("canvas");
  canvas.width = exp.width; canvas.height = exp.height;
  var ctx = canvas.getContext("2d");
  var timestamp = 0;
  for (var i = 0; i < exp.frames.length; i++) {
    ctx.putImageData(exp.frames[i].imageData, 0, 0);
    var durationUs = Math.max(1, Math.round(exp.frames[i].delayMs * 1000));
    var frame = new VideoFrame(canvas, { timestamp: timestamp, duration: durationUs, alpha: "keep" });
    encoder.encode(frame, { keyFrame: i === 0 });
    frame.close();
    timestamp += durationUs;
    if (statusPrefix) downloadStatus.textContent = statusPrefix + " " + (i + 1) + " / " + exp.frames.length;
    if (encoder.encodeQueueSize > 4) {
      await new Promise(function (r) {
        var t = setInterval(function () { if (encoder.encodeQueueSize <= 4) { clearInterval(t); r(); } }, 30);
      });
    }
  }
  await encoder.flush();
  if (encoderError) throw encoderError;
  muxer.finalize();
  return new Blob([muxer.target.buffer], { type: "video/webm" });
}

downloadWebmBtn.addEventListener("click", async function () {
  var item = activeItem();
  if (!item || !item.outputs || !item.outputs.length || !webmSupported) return;
  downloadWebmBtn.disabled = true;
  downloadStatus.textContent = "Encoding the WebM...";
  try {
    var blob = await encodeWebmBlob(item, "Encoding the WebM...");
    downloadBlob(blob, item.baseName + "-cutout-alpha.webm");
    downloadStatus.textContent = "WebM downloaded — VP9 with a real alpha channel.";
  } catch (e) {
    downloadStatus.textContent = "WebM encoding failed in this browser (" + (e.message || e) + "). The PNG sequence above always works.";
  }
  downloadWebmBtn.disabled = false;
});

/* Batch download: every finished item, in the format picked by the OUTPUT
 * chips, each in its own folder inside one .zip. If WebM is picked but a
 * file's dimensions are not encodable here, that file falls back to GIF
 * and the status line says so. */
downloadAllBtn.addEventListener("click", async function () {
  var done = state.queue.filter(function (it) { return it.status === "done" && it.outputs && it.outputs.length; });
  if (done.length < 2 || !window.JSZip) return;
  downloadAllBtn.disabled = true;
  var fellBack = false;
  try {
    var zip = new JSZip();
    for (var i = 0; i < done.length; i++) {
      var item = done[i];
      var folder = zip.folder(item.baseName);
      downloadStatus.textContent = "Packing file " + (i + 1) + " of " + done.length + " (" + item.name + ")...";
      if (state.outputFormat === "gif") {
        folder.file(item.baseName + "-cutout.gif", await encodeGifBlob(item, null));
      } else if (state.outputFormat === "webm") {
        var dims = exportDimsFor(item);
        if (await webmConfigOk(dims.w, dims.h)) {
          folder.file(item.baseName + "-cutout-alpha.webm", await encodeWebmBlob(item, null));
        } else {
          folder.file(item.baseName + "-cutout.gif", await encodeGifBlob(item, null));
          fellBack = true;
        }
      } else {
        await addPngFramesToZip(folder, item);
      }
      await yieldToUI();
    }
    var out = await zip.generateAsync({ type: "blob", compression: "STORE" });
    downloadBlob(out, "snipgif-batch-" + done.length + "-files.zip");
    downloadStatus.textContent = "Batch ZIP downloaded — " + done.length + " files, one folder each" +
      (fellBack ? " (some files fell back to GIF: WebM is not encodable at their size here)." : ".");
  } catch (e) {
    downloadStatus.textContent = "Batch ZIP failed: " + (e.message || e);
  }
  downloadAllBtn.disabled = false;
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
  var files = e.dataTransfer && e.dataTransfer.files;
  if (files && files.length) addFiles(files);
});
fileInput.addEventListener("change", function () {
  if (fileInput.files && fileInput.files.length) addFiles(fileInput.files);
  fileInput.value = "";
});
processBtn.addEventListener("click", processAll);

function syncPreviewBackdrop() {
  var checker = previewSelect.value === "checker";
  afterBox.classList.toggle("sg-checker", checker);
  if (afterLabel) {
    afterLabel.textContent = checker ? "AFTER — TRANSPARENT (ON CHECKERBOARD)" : "AFTER — TRANSPARENT (SHOWN ON BLACK)";
  }
}
previewSelect.addEventListener("change", syncPreviewBackdrop);
syncPreviewBackdrop();

function setOutputChoice(format) {
  state.outputFormat = format;
  var map = { png: chipPng, webm: chipWebm, gif: chipGif };
  Object.keys(map).forEach(function (k) {
    map[k].classList.toggle("sg-chip-on", k === format);
    map[k].setAttribute("aria-checked", String(k === format));
  });
  downloadPngBtn.classList.toggle("sg-primary-choice", format === "png");
  downloadWebmBtn.classList.toggle("sg-primary-choice", format === "webm");
  downloadGifBtn.classList.toggle("sg-primary-choice", format === "gif");
}
chipPng.addEventListener("click", function () { setOutputChoice("png"); });
chipWebm.addEventListener("click", function () { setOutputChoice("webm"); });
chipGif.addEventListener("click", function () { setOutputChoice("gif"); });
setOutputChoice("gif");

function setEffect(name) {
  state.effect = name;
  Object.keys(fxChips).forEach(function (k) {
    fxChips[k].classList.toggle("sg-chip-on", k === name);
    fxChips[k].setAttribute("aria-checked", String(k === name));
  });
  onFxChanged();
}
Object.keys(fxChips).forEach(function (k) {
  fxChips[k].addEventListener("click", function () { setEffect(k); });
});
fxIntensity.addEventListener("input", function () {
  state.intensity = Number(fxIntensity.value) || 5;
  fxIntensityVal.textContent = String(state.intensity);
  onFxChanged();
});
fxIntensityVal.textContent = String(state.intensity);
renderQueue();
updateProcessBtn();
