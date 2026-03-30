# Claude Protocol Inspector (SSE Stream Inspector)

## Overview

A teaching tool for analyzing and visualizing Claude AI dialogue protocols. It parses raw SSE (Server-Sent Events) streams and Claude HTTP JSON dialogue histories, then renders them as structured, human-readable visualizations.

## Tech Stack

- **Language**: TypeScript
- **Framework**: React 19
- **Build Tool**: Vite 6
- **Styling**: Tailwind CSS (loaded via CDN)
- **Fonts**: Inter (UI) + Fira Code (monospace)
- **Package Manager**: npm

## Project Structure

```
util/
├── App.tsx              # Main app component with input/output layout
├── index.tsx            # React entry point
├── index.html           # HTML shell with Tailwind CDN and importmap
├── types.ts             # TypeScript interfaces (SSEEvent, MessageState, ClaudeChatHistory, etc.)
├── components/
│   ├── EventItem.tsx    # Renders individual SSE events
│   ├── MessagePreview.tsx  # Reconstructed message view for SSE mode
│   └── ChatHistory.tsx  # Full dialogue history view for JSON mode
├── services/
│   └── sseParser.ts     # SSE text parsing and message reconstruction logic
├── vite.config.ts       # Vite config (port 3000, path alias @/)
├── tsconfig.json        # TypeScript config (ES2022, React JSX)
├── package.json         # Dependencies and scripts
└── .env.local           # Environment variables (GEMINI_API_KEY)
```

## Functionality

1. **SSE Stream Parsing**: Paste raw SSE text (including HTTP headers) to parse into individual events (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_stop`).
2. **Dialogue JSON Parsing**: Paste Claude API HTTP JSON (with `messages` array) to visualize multi-turn conversations including thinking blocks, tool use, and tool results.
3. **Auto-Detection**: Automatically detects whether input is SSE or JSON and switches view mode accordingly.
4. **Built-in Examples**: Includes preloaded SSE and dialogue examples for quick demo.

## Build and Run

### Prerequisites

- Node.js (v18+)
- npm

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

App runs at **http://localhost:3000**.

### Build for Production

```bash
npm run build
```

Output goes to `dist/`.

### Preview Production Build

```bash
npm run preview
```

## Environment Variables

Create a `.env.local` file (already exists) with:

```
GEMINI_API_KEY=your_key_here
```

Note: The Gemini API key is defined in vite config but not currently used by the app's core functionality.
