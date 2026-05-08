# GYSTC — Instructions for Claude

> Copy this into your CLAUDE.md or Claude Code project settings.
> Adjust the region table if you've customized your brain regions.

## Your Brain — the Obsidian Vault

You have a persistent long-term memory via the GYSTC MCP server. Your vault is your brain — use it.

**Core principle:** Search the vault BEFORE answering from general knowledge. If the user asks about a project, decision, or past work — the vault likely has it. Don't wait to be told to search.

## When to Search (Proactively)

- User mentions a project name → `brain_retrieve("project name")`
- User asks "how did we do X?" → `brain_retrieve("X")`
- You're about to suggest an approach → check for prior decisions first
- Starting a new task → `brain_context(task_description="...")`
- User references something from a past session → it's in the vault

## When to Store

- You discover an important architectural decision
- User shares context that future sessions will need
- A project plan or design is finalized
- You learn something non-obvious about the user's setup

## The 12 Brain Regions

Notes are organized by FUNCTION, not topic:

| Idx | Region | Responsibility |
|-----|--------|----------------|
| 0 | Praefrontaler Cortex | Architecture, decisions, planning |
| 1 | Motorischer Cortex | API writes, actions, execution |
| 2 | Sensorischer Cortex | Data intake, references, input |
| 3 | Hippocampus | Memory, sessions, personal notes |
| 4 | Kleinhirn | Precision algorithms |
| 5 | Nucleus Accumbens | Subscriptions, pricing, rewards |
| 6 | Broca-Areal | AI, prompts, agents, language |
| 7 | Visueller Cortex | UI, themes, design |
| 8 | Thalamus | Index, MOC, data relay |
| 9 | Stammhirn | Config, infrastructure (default) |
| 10 | Basalganglien | Pipelines, ETL, background jobs |
| 11 | Amygdala | Auth, team, social interaction |

## Tool Quick Reference

| Tool | Use When |
|------|----------|
| `brain_retrieve` | Searching for knowledge by topic or question |
| `brain_context` | Loading context for current work (files + task) |
| `brain_store` | Saving important knowledge for the future |
| `brain_recent` | Catching up on recent vault changes |
| `brain_related` | Exploring connections from a specific note |
| `brain_history` | Checking version history of a note |
| `brain_diff` | Comparing current vs previous version |
| `brain_rollback` | Restoring a note to a previous version |
| `brain_status` | Health check (instant, no model needed) |
| `brain_classify` | Auto-classify notes into brain regions |
| `brain_reclassify` | Re-classify a misclassified note |
| `brain_autolink` | Find and create missing backlinks |
| `brain_enrich` | Add metadata/tags to sparse notes |
| `brain_reindex` | Re-index after bulk changes |
| `brain_regions` | List all 12 regions with note counts |

## Session-Start Hook

If the hook is installed, every new Claude Code session automatically gets vault context
injected (recent changes, relevant notes based on git context). This runs via FTS5-only
search in under 100ms. You'll see the context at the start of each conversation.

## Behavior

- Search proactively — don't wait for explicit commands
- Summarize findings — don't dump raw note content
- Reference paths — so the user can dig deeper in Obsidian
- Store sparingly — only knowledge worth keeping across sessions
- Use regions when storing — helps with organization and filtered search
- The Dashboard (GYSTC Dashboard.exe) visualizes your vault as a 3D brain
