# GYSTC Landing Page — Prompt für Claude Design

## Projekt

Erstelle eine **Landing Page** für "Get Your Shit Together, Claude." (GYSTC).

## Designsprache

**Brutalistisch.** Keine Gradients, keine Animationen, keine Stock-Fotos, keine AI-Buzzwords. Raw, ehrlich, direkt. Denk an: Monospace-Typografie, dunkler Hintergrund (#0a0a0a oder ähnlich), harter Kontrast, viel Whitespace, industrielle Ästhetik. Inspiration: brutalistwebsites.com — aber polished genug dass es professional wirkt, nicht wie ein Studentenprojekt.

## Farben

- Background: #0a0a0a
- Text: #e0e0e0
- Akzent für CTAs: hartes Weiss oder ein einzelner Farbton — kein Gradient
- Monochrom bis auf die CTA-Buttons

## Typografie

Monospace als Hauptschrift (JetBrains Mono, IBM Plex Mono, oder Source Code Pro). Grosses Font-Weight für Headlines.

## Was das Produkt ist

GYSTC ist ein MCP-Server (Model Context Protocol) für Claude. Das Problem: Claude vergisst alles nach jeder Session. Jedes Mal fängt man bei Null an — egal wie lange man zusammengearbeitet hat. GYSTC gibt Claude ein persistentes Langzeitgedächtnis über Sessions, Projekte und Monate hinweg. Es ist KEIN Gehirn-Simulator — es ist ein praktisches Dev-Tool das Claude besser macht.

## Technologie (kurz erwähnen, nicht zu deep)

- Semantic Search (FAISS + FTS5 Hybrid)
- Cross-Encoder Re-Ranking für Präzision
- Smart Chunking für lange Dokumente
- Content-Addressable Storage mit Versioning
- Auto-Context: Claude bekommt beim Start automatisch relevanten Kontext
- 12 Wissensregionen für strukturierte Organisation

## Sections der Page (in dieser Reihenfolge)

### 1. Hero — Gross, brutal, direkt.

- Headline: "Get Your Shit Together, Claude."
- Subline: "Your AI forgets everything. Every. Single. Session. This fixes that."
- Zwei Buttons nebeneinander: [DOWNLOAD — free] und [SOURCE CODE — via Patreon]

### 2. Das Problem — Kurzer, harter Text.

- "You've explained your project structure to Claude 50 times. It forgets your architecture decisions. It doesn't know your conventions. Every session is Groundhog Day."
- Kein Bild, nur Text. Monospace. Brutal ehrlich.

### 3. Wie es funktioniert — 3 Steps, minimalistisch.

- Install → Configure Vault Path → Claude remembers everything.
- Jeder Step maximal 1 Satz. Keine langen Erklärungen.

### 4. Vorher/Nachher — Zwei Spalten, harter Kontrast.

- Links (Before): "Session 47: 'Can you explain your project structure again?'"
- Rechts (After): "Session 47: 'Based on your architecture decisions from March, I'd suggest...'"
- Monospace, code-block Ästhetik.

### 5. Features — Keine Feature-Cards, keine Icons. Einfach eine Liste.

Bullet Points, Monospace, links-aligned. Jedes Feature ein Satz:

- Persistent memory across sessions and projects
- Semantic search (FAISS vector similarity)
- Full-text search (SQLite FTS5)
- Hybrid search with Reciprocal Rank Fusion
- Cross-Encoder Re-Ranking for precision
- 12 knowledge regions for structured organization
- Segmental search — queries target relevant regions
- Smart chunking for long documents
- Content versioning with rollback
- Auto-context on session start
- Background model loading — zero blocking

### 6. Tech Stack — Für die Devs die es wissen wollen.

- Python, FAISS, SQLite FTS5, SentenceTransformers, FastMCP
- Keine Logos, nur Text.

### 7. Download Section — Repeat CTA.

- [DOWNLOAD EXE — $0] — Gross, prominent
- [SOURCE CODE — Patreon] — Daneben, gleich gross
- "The EXE is and always will be free. Source code access via Patreon supports future projects."

### 8. Footer — Minimal.

- "Built by Lucas." + Patreon Link + GitHub Link (für Issues)

## Was NICHT auf die Page soll

- Keine generischen AI-Brain Stockfotos (leuchtende Neuronen, Synapsen-Grafiken, etc.)
- Keine Testimonials (noch keine User)
- Keine Pricing-Tabelle (es gibt nur free + Patreon)
- Keine Animations oder Hover-Effekte
- Keine "Trusted by" Logos
- Keine Newsletter-Signup
- Kein Cookie-Banner

## Was erwähnt werden SOLL

- Das System ist intern in 12 Wissensregionen organisiert (wie Hirnareale: Architektur, Config, UI, Auth, etc.)
- Segmental Search: Claude sucht gezielt in der relevanten Region statt alles zu durchsuchen
- Die Struktur ist visuell wie ein Gehirn aufgebaut — aber das Ziel ist Struktur und Performance, nicht "wir simulieren ein Gehirn"
- Ton: "Organized like a brain. Works like a search engine. Performs like neither existed before."

## Bildmaterial

Platzhalter für Screenshots aus der echten App (werden später eingefügt). Markiere die Stellen mit [SCREENSHOT PLACEHOLDER].

## Tone of Voice

Direkt, leicht frech, technical, kein Marketing-Sprech. So wie ein Dev mit einem anderen Dev redet — nicht wie ein SaaS-Landing-Page mit einem "Enterprise Decision Maker".

## Monetization-Modell

- EXE Download: Kostenlos für jeden, kein Account nötig
- Source Code / Open Source: Nur über Patreon-Payment zugänglich
- Patreon-Mitglieder bekommen: Zugang zum Source Code, Insights, weitere Projekte
- Das Produkt selbst (die EXE) bleibt gratis
