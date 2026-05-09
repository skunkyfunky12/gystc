# GYSTC Reddit Posts

---

## r/ClaudeAI

**Title:** I built a local MCP server that gives Claude persistent memory across sessions — open source, free forever

**Body:**

I got tired of re-explaining my entire project context every session. So I built GYSTC (Get Your Shit Together, Claude) — a local MCP server that turns your Obsidian vault into Claude's long-term memory.

**What it does:**
- Claude automatically searches your vault before answering from general knowledge
- Hybrid search: FAISS vectors + FTS5 keywords + cross-encoder re-ranking
- 16 MCP tools Claude calls autonomously — no slash commands needed
- Auto-context injection on every session start via hooks
- 12 "brain regions" for structured knowledge organization

**What it doesn't do:**
- No cloud. No telemetry. No account. Everything runs on your machine.

Comes with a free desktop app (Windows + macOS) that visualizes your knowledge graph as a 3D brain — mostly for fun, but it's genuinely useful for spotting gaps in your notes.

Built this solo over the last few weeks. Would love feedback.

🌐 https://gystc.dev
📦 https://github.com/skunkyfunky12/gystc

---

## r/ObsidianMD

**Title:** Turned my Obsidian vault into Claude's brain — free MCP server that gives AI persistent memory via your notes

**Body:**

If you use both Obsidian and Claude Code, this might be useful: I built an MCP server that lets Claude read, search, and write to your Obsidian vault automatically.

**How it works with Obsidian:**
- Your vault becomes Claude's long-term memory — organized into 12 "brain regions" mapped to your folder structure
- Hybrid search across all your notes (semantic via FAISS + keyword via FTS5)
- Claude can store new knowledge as notes, create backlinks between them, and auto-classify them into the right region
- Backlinks = synaptic connections. Claude follows them to find related context.
- Content versioning with rollback — so Claude can't accidentally destroy your notes

**Setup:** One-click install via the setup wizard. Works alongside your existing vault — doesn't move or restructure anything. Reads .md files, respects your folder structure.

The desktop app also has a 3D knowledge graph visualization with click-to-open-in-Obsidian via the Local REST API plugin.

100% local, free, open source.

🌐 https://gystc.dev
📦 https://github.com/skunkyfunky12/gystc

---

## r/LocalLLaMA

**Title:** Open source MCP server for persistent AI memory — local-first, FAISS + FTS5 hybrid search, works with Obsidian vaults

**Body:**

Built a local MCP server that gives Claude persistent long-term memory using your Obsidian vault as the knowledge store. Sharing because the architecture might be interesting even if you don't use Claude.

**Tech stack:**
- **Storage:** SQLite with FTS5 for full-text search + content versioning
- **Vectors:** sentence-transformers embeddings → FAISS index (384-dim, all-MiniLM-L6-v2)
- **Search:** Hybrid retrieval via Reciprocal Rank Fusion (semantic + keyword), then cross-encoder re-ranking
- **Chunking:** Smart chunking for long documents with overlap
- **Protocol:** FastMCP — 16 tools exposed via Model Context Protocol
- **Dashboard:** PyQt6 + Three.js 3D graph visualization (WebEngine)
- **Indexing:** Background watcher via watchdog, zero-blocking startup with async model loading

The vault is partitioned into 12 "brain regions" — queries get routed to relevant regions instead of flat-scanning everything. There's also an auto-classifier that learns from corrections.

159 tests, ~2s full suite. Free desktop app for Windows + macOS.

Not tied to Claude specifically — the MCP protocol is open, so this could work with any MCP-compatible client.

🌐 https://gystc.dev
📦 https://github.com/skunkyfunky12/gystc
