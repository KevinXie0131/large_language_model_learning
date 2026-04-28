# yt-audio-downloader

CLI tool that downloads audio from YouTube videos. The Node entrypoint (`index.js`) shells out to `yt-dlp` (Python) and optionally `ffmpeg` for MP3 conversion. Output goes to `./downloads/`.

## Prerequisites

- **Node.js** (any recent LTS)
- **Python 3** — required; invoked as `python` on PATH
- **yt-dlp** — installed as a Python module (`python -m yt_dlp`)
  - `pip install -U yt-dlp`
- **ffmpeg** (optional) — if present on PATH, downloads are converted to MP3 at best quality; otherwise the original audio format is kept

## Install

```bash
npm install
```

## Run

Three modes, all via `node index.js` (or `npm start` / `npm run download`):

1. **Interactive prompt** — no args:
   ```bash
   node index.js
   ```
   Paste a YouTube URL at the prompt; `q` to quit.

2. **Single URL**:
   ```bash
   node index.js "https://www.youtube.com/watch?v=VIDEO_ID"
   ```

3. **Batch from .txt file** — one URL per line, `#` for comments:
   ```bash
   node index.js links.txt
   ```
   Invalid lines are skipped. Playlists are not expanded (`--no-playlist`).

## Output

Files are written to `downloads/` as `<video title>.<ext>` (mp3 if ffmpeg is available, otherwise the source format, e.g. m4a/webm).

## Troubleshooting

- `Failed to run yt-dlp: ...` — Python isn't on PATH, or `yt_dlp` isn't installed. Run `python -m yt_dlp --version` to verify.
- Audio isn't MP3 — ffmpeg isn't on PATH. Install it and re-run; the script auto-detects it at startup.
- URL rejected as invalid — only `youtube.com/watch`, `youtu.be/`, and `youtube.com/shorts/` URLs with an `http(s)://` scheme are accepted (see `isValidURL` in `index.js:17`).
