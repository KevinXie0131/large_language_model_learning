const { spawn, execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const OUTPUT_DIR = path.join(__dirname, "downloads");
const DEFAULT_LINKS_FILE = path.join(__dirname, "links.txt");

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

function downloadVideo(url) {
  return new Promise((resolve, reject) => {
    console.log(`\nDownloading: ${url}`);

    const args = [
      "-m", "yt_dlp",
      "-o", path.join(OUTPUT_DIR, "%(title)s.%(ext)s"),
      "--no-playlist",
      "--progress",
    ];

    if (HAS_FFMPEG) {
      args.push("-f", "bv*+ba/b", "--merge-output-format", "mp4");
      console.log("(ffmpeg found — best video+audio merged to mp4)");
    } else {
      args.push("-f", "b");
      console.log("(no ffmpeg — saving best single-stream format)");
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

  console.log(`Reading from: ${resolved}`);
  console.log(`Found ${urls.length} URL(s) in file.`);
  if (skipped > 0) {
    console.log(`Skipped ${skipped} invalid line(s).`);
  }

  let success = 0;
  let failed = 0;

  for (let i = 0; i < urls.length; i++) {
    console.log(`\n--- [${i + 1}/${urls.length}] ---`);
    try {
      await downloadVideo(urls[i]);
      success++;
    } catch (err) {
      console.error(`Failed: ${err.message}`);
      failed++;
    }
  }

  console.log(`\n=== Done: ${success} downloaded, ${failed} failed ===`);
}

function printUsage() {
  console.log("Usage:");
  console.log("  node index.js                        # read URLs from ./links.txt");
  console.log("  node index.js <path/to/links.txt>    # read URLs from a custom file");
  console.log("  node index.js <youtube-url>          # download a single video");
}

ensureOutputDir();

const arg = process.argv[2];

if (arg) {
  if (isValidURL(arg)) {
    downloadVideo(arg)
      .then(() => process.exit(0))
      .catch((err) => {
        console.error(`Failed: ${err.message}`);
        process.exit(1);
      });
  } else {
    downloadFromFile(arg).then(() => process.exit(0));
  }
} else {
  if (fs.existsSync(DEFAULT_LINKS_FILE)) {
    downloadFromFile(DEFAULT_LINKS_FILE).then(() => process.exit(0));
  } else {
    console.error(`No argument given and ${DEFAULT_LINKS_FILE} does not exist.\n`);
    printUsage();
    process.exit(1);
  }
}
