# Optional: self-hosted model chunks

This folder is empty in the standalone repo — and that is fine. On first
use, `snipgif.js` tries to load each AI model from chunked files here,
and if they are missing it automatically falls back to the Hugging Face
mirrors of the official rembg models.

To self-host (like the live Blob site does):

1. Download `u2net.onnx` and `isnet-general-use.onnx` from
   https://huggingface.co/tomjackson2023/rembg (or the fallback mirror
   https://huggingface.co/lucasfaesa/rembg-models).
2. Split each file into 20 MiB (20,971,520-byte) parts — 9 parts each:
   - `split -b 20971520 -d u2net.onnx u2net.part`
   - `split -b 20971520 -d isnet-general-use.onnx isnet.part`
3. Put `u2net.part00`…`u2net.part08` and `isnet.part00`…`isnet.part08`
   in this folder. No code change needed.

Expected reassembled sizes / MD5s (identical to rembg's official
release):

- u2net: 175,997,641 bytes, MD5 `60024c5c889badc19c04ad937298a77b`
- isnet: 178,648,008 bytes, MD5 `fc16ebd8b0c10d971d3513d564d01e29`
