const { spawn, execSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const readline = require("readline");

const OUTPUT_DIR = path.join(__dirname, "downloads");

const HAS_FFMPEG = (() => {
  try {
    execSync("ffmpeg -version", { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
})();

function isValidURL(url) {
  return /^https?:\/\/(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/shorts\/)/.test(url);
}

function ensureOutputDir() {
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR);
  }
}

function downloadAudio(url) {
  return new Promise((resolve, reject) => {
    console.log(`\nDownloading: ${url}`);

    const args = [
      "-m", "yt_dlp",
      "-x",                          // extract audio
      "-o", path.join(OUTPUT_DIR, "%(title)s.%(ext)s"),
      "--no-playlist",               // single video only
      "--progress",
    ];

    if (HAS_FFMPEG) {
      args.push("--audio-format", "mp3", "--audio-quality", "0");
      console.log("(ffmpeg found — converting to mp3)");
    } else {
      console.log("(no ffmpeg — saving original audio format)");
    }

    args.push(url);

    const proc = spawn("python", args, { stdio: "inherit" });

    proc.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`yt-dlp exited with code ${code}`));
      }
    });

    proc.on("error", (err) => {
      reject(new Error(`Failed to run yt-dlp: ${err.message}`));
    });
  });
}

async function downloadFromFile(filePath) {
  const resolved = path.resolve(filePath);

  if (!fs.existsSync(resolved)) {
    console.error(`File not found: ${resolved}`);
    process.exit(1);
  }

  const content = fs.readFileSync(resolved, "utf-8");
  const lines = content
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));

  const urls = lines.filter((l) => isValidURL(l));
  const skipped = lines.length - urls.length;

  if (urls.length === 0) {
    console.error("No valid YouTube URLs found in the file.");
    process.exit(1);
  }

  console.log(`Found ${urls.length} URL(s) in file.`);
  if (skipped > 0) {
    console.log(`Skipped ${skipped} invalid line(s).`);
  }

  let success = 0;
  let failed = 0;

  for (let i = 0; i < urls.length; i++) {
    console.log(`\n--- [${i + 1}/${urls.length}] ---`);
    try {
      await downloadAudio(urls[i]);
      success++;
    } catch (err) {
      console.error(`Failed: ${err.message}`);
      failed++;
    }
  }

  console.log(`\n=== Done: ${success} downloaded, ${failed} failed ===`);
}

function prompt() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  function ask() {
    rl.question('\nPaste a YouTube URL (or "q" to quit): ', async (input) => {
      const url = input.trim();
      if (url.toLowerCase() === "q") {
        console.log("Bye!");
        rl.close();
        return;
      }

      if (!isValidURL(url)) {
        console.error("That doesn't look like a valid YouTube URL. Try again.");
        ask();
        return;
      }

      try {
        await downloadAudio(url);
      } catch (err) {
        console.error(`Failed: ${err.message}`);
      }
      ask();
    });
  }

  ask();
}

// --- Main ---
ensureOutputDir();

const arg = process.argv[2];
if (arg) {
  if (arg.endsWith(".txt")) {
    downloadFromFile(arg).then(() => process.exit(0));
  } else if (isValidURL(arg)) {
    downloadAudio(arg)
      .then(() => process.exit(0))
      .catch((err) => {
        console.error(`Failed: ${err.message}`);
        process.exit(1);
      });
  } else {
    console.error("Invalid input. Provide a YouTube URL or a .txt file path.");
    process.exit(1);
  }
} else {
  console.log("=== YouTube Audio Downloader ===");
  console.log('Tip: pass a .txt file with URLs to batch download.');
  prompt();
}
