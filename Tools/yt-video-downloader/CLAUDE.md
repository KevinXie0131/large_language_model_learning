# yt-video-downloader

CLI tool that downloads YouTube videos from URLs listed in a text file. The Node entrypoint (`index.js`) shells out to `yt-dlp` (Python module) and uses `ffmpeg` to merge bestvideo+bestaudio streams into an `.mp4`. Output goes to `./downloads/`.

## Tech stack

| Layer        | Tool / Library | Version (verified locally) | Purpose                                                                     |
| ------------ | -------------- | -------------------------- | --------------------------------------------------------------------------- |
| Runtime      | Node.js        | v22.22.2                   | Runs `index.js`, parses CLI args, orchestrates the download loop            |
| Language     | JavaScript (CommonJS) | ES2022                  | Single-file script using `require(...)`                                     |
| Node std lib | `child_process`, `fs`, `path` | built-in     | `spawn`/`execSync` to call yt-dlp; sync file I/O; cross-platform paths      |
| Downloader   | yt-dlp         | 2026.03.17                 | Actual YouTube download + format selection (Python module, `python -m yt_dlp`) |
| Python       | CPython        | 3.12.10                    | Hosts the yt-dlp module                                                     |
| Muxer        | ffmpeg         | 8.1                        | Merges bestvideo + bestaudio streams into a single `.mp4`                   |

No third-party Node packages — `package.json` declares no `dependencies`.

### Framework

None. This is a plain Node CLI script (no Express, no Commander, no build step). CLI dispatch is a hand-rolled `process.argv[2]` switch in `index.js`.

## Prerequisites

Install once on the host:

1. **Node.js ≥ 18** (tested on v22.22.2) — https://nodejs.org/
2. **Python 3** on PATH (tested on 3.12.10) — invoked as `python`
3. **yt-dlp** as a Python module:
   ```bash
   pip install -U yt-dlp
   ```
   Verify: `python -m yt_dlp --version`
4. **ffmpeg** on PATH (tested on 8.1). If absent, the tool still runs but falls back to a single combined stream (no merge) and warns at startup.
   Verify: `ffmpeg -version`

## Build

There is no build step. Optionally:

```bash
cd C:\Backup\Node\yt-video-downloader
npm install
```

This is a no-op for runtime (no deps) but creates `package-lock.json`.

## Run locally

From `C:\Backup\Node\yt-video-downloader`:

1. **Batch from default `links.txt`** (one URL per line, `#` for comments):
   ```bash
   node index.js
   ```
   Equivalent: `npm start` or `npm run download`.

2. **Batch from a custom file**:
   ```bash
   node index.js path/to/other-list.txt
   ```

3. **Single URL**:
   ```bash
   node index.js "https://www.youtube.com/watch?v=VIDEO_ID"
   ```

Accepted URL forms (see `isValidURL` in `index.js`): `youtube.com/watch?v=...`, `youtu.be/...`, `youtube.com/shorts/...` with `http(s)://` scheme. Playlists are not expanded (`--no-playlist`).

## Output

Files land in `./downloads/` named `<video title>.mp4` (or the source container if ffmpeg is unavailable, e.g. `.webm`/`.mp4` single stream).

## Format selection

- ffmpeg present → `-f "bv*+ba/b" --merge-output-format mp4` (best video + best audio, merged)
- ffmpeg absent → `-f b` (best single combined stream, no merge)

## Troubleshooting

- `Failed to run yt-dlp: ...` — `python` isn't on PATH. Run `python --version`.
- `No module named yt_dlp` — install it: `pip install -U yt-dlp`.
- Video isn't merged / lower quality than expected — ffmpeg not on PATH; install it and rerun.
- `URL rejected as invalid` — only `youtube.com/watch`, `youtu.be/`, `youtube.com/shorts/` URLs are accepted.
- `No valid YouTube URLs found in the file` — `links.txt` only contains comments/blank lines.
