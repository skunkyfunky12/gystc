# Neural Brain — Classification Specification

Portable spec for classifying Obsidian vault notes into 12 brain regions.
Any classifier (human, keyword engine, or LLM) should follow this decision tree.

---

## The 12 Brain Regions

| Idx | Region | Slug | Function | Example Content |
|-----|--------|------|----------|-----------------|
| 0 | Präfrontaler Cortex | `praefrontaler-cortex` | Architecture, decisions, planning | RFCs, roadmaps, design docs, project specs, audit findings |
| 1 | Motorischer Cortex | `motorischer-cortex` | Execution, writes, deploys, code actions | Commits, migrations, endpoints, CRUD, deploy scripts, API writes |
| 2 | Sensorischer Cortex | `sensorischer-cortex` | Data intake, monitoring, observation | Logs, metrics, Grafana, Sentry, analytics, data sources, census data |
| 3 | Hippocampus | `hippocampus` | Memory, sessions, personal notes | Session logs, journals, lessons learned, feedback, personal reflections |
| 4 | Kleinhirn | `kleinhirn` | Precision algorithms, computation | Scoring formulas, distance calculations, parsers, regex, heuristics |
| 5 | Nucleus Accumbens | `nucleus-accumbens` | Monetization, pricing, rewards | Subscription tiers, Stripe billing, pricing models, revenue, coupons |
| 6 | Broca-Areal | `broca-areal` | Language generation, AI, prompts | LLM prompts, agents, embeddings, RAG, MCP tools, NLP |
| 7 | Visueller Cortex | `visueller-cortex` | UI, rendering, visual design | CSS, themes, components, layouts, animations, frontend, Figma, SVG |
| 8 | Thalamus | `thalamus` | Routing, indexing, relay | MOC files, navigation, registries, hub pages, sitemaps |
| 9 | Stammhirn | `stammhirn` | Config, infrastructure, system | Docker, nginx, DNS, env vars, dotfiles, system setup |
| 10 | Basalganglien | `basalganglien` | Orchestration, pipelines, scheduling | CI/CD, GitHub Actions, queues, workers, watchers, ETL, linting |
| 11 | Amygdala | `amygdala` | Auth, security, social/team | JWT, OAuth, RLS, permissions, RBAC, team invites, vulnerability fixes |

---

## Decision Tree: FUNKTION > FORM > THEMA

### Layer 0: Explicit Tag (instant)
If the note contains `#brain/<slug>`, return that region. No further analysis.

### Layer 1: FUNKTION — What does this content DO?

Ask: "What is the primary *action* or *purpose* of this content?"

```
Plans / decides / architects        → 0  Präfrontaler Cortex
Executes / writes / deploys / ships → 1  Motorischer Cortex
Monitors / observes / logs / reports→ 2  Sensorischer Cortex
Remembers / reflects / learns       → 3  Hippocampus
Computes precisely / calculates     → 4  Kleinhirn
Monetizes / prices / bills          → 5  Nucleus Accumbens
Generates language / prompts / AI   → 6  Broca-Areal
Displays / renders / styles         → 7  Visueller Cortex
Routes / indexes / relays           → 8  Thalamus
Configures / sets up infrastructure → 9  Stammhirn
Orchestrates / schedules / pipelines→ 10 Basalganglien
Authenticates / secures / guards    → 11 Amygdala
```

If FUNKTION clearly resolves → done.

### Layer 2: FORM — What type of artifact is this?

Used as tiebreaker when FUNKTION is ambiguous.

```
RFC / design doc / proposal         → 0
Migration / deploy script           → 1
Dashboard / metrics page            → 2
Session log / journal               → 3
Algorithm / formula / scoring       → 4
Pricing table / billing config      → 5
Prompt template / agent definition  → 6
Component / stylesheet / theme      → 7
Index page / MOC / registry         → 8
Config file / Dockerfile / .env     → 9
CI/CD workflow / cron / watcher     → 10
Auth middleware / policy / RLS      → 11
```

### Layer 3: THEMA — What domain is this about?

Lowest priority. Only when FUNKTION + FORM don't resolve.

```
Project management domain           → 0
Database / API domain               → 1
Analytics / geodata domain          → 2
Personal / biographical domain      → 3
Math / statistics domain            → 4
Business / SaaS domain              → 5
NLP / LLM / AI domain              → 6
Design / CSS / UI domain            → 7
Navigation / information arch.      → 8
DevOps / infrastructure domain     → 9
Build / test / CI domain            → 10
Security / compliance domain        → 11
```

---

## Disambiguation Rules

These resolve common conflicts:

| Conflict | Rule | Reason |
|----------|------|--------|
| Kleinhirn (4) vs Basalganglien (10) | If `pipeline`, `ci`, `schedule`, `queue`, `worker`, `cron`, `watcher` → 10. If `algorithm`, `formula`, `scoring`, `calculation`, `heuristic` → 4 | Kleinhirn = **computes**. Basalganglien = **orchestrates**. |
| Motorischer Cortex (1) vs Basalganglien (10) | If the content *does* the action → 1. If it *schedules/chains* actions → 10 | Motor = execution. Basal = orchestration of execution. |
| Sensorischer Cortex (2) vs Thalamus (8) | If content *collects/displays* data → 2. If it *routes/indexes* data → 8 | Sensory = intake. Thalamus = relay. |
| Broca-Areal (6) vs Präfrontaler Cortex (0) | If about *how to prompt/generate* → 6. If about *what to build/decide* → 0 | Broca = language production. Prefrontal = planning. |
| Stammhirn (9) vs Motorischer Cortex (1) | If *setting up* infrastructure → 9. If *executing* a deploy/migration → 1 | Stammhirn = config. Motor = action. |
| Visueller Cortex (7) vs Motorischer Cortex (1) | If about *how it looks* → 7. If about *building/shipping* it → 1 | Visual = rendering. Motor = execution. |

---

## Example Classifications

### Example 1: "OSRM Routing Performance Tuning"
- FUNKTION: Computes routes precisely → Kleinhirn? Or configures OSRM → Stammhirn?
- FORM: Config file with algorithm parameters → tiebreaker needed
- Content mentions: `docker`, `nginx`, `osrm-extract`, `profiles` → infrastructure setup
- **Result: 9 Stammhirn** (configuring/setting up OSRM, not computing routes)

### Example 2: "scoring-engine.ts"
- FUNKTION: Computes scores precisely
- Content: `weight`, `formula`, `distance`, `heuristic`, `score = ...`
- **Result: 4 Kleinhirn** (precision computation)

### Example 3: "GitHub Actions CI Pipeline"
- FUNKTION: Orchestrates build/test/deploy
- Content: `workflow`, `on: push`, `jobs:`, `steps:`
- **Result: 10 Basalganglien** (pipeline orchestration)

### Example 4: "Stripe Webhook Handler"
- FUNKTION: Receives payment events → Motorischer Cortex (1)? Nucleus Accumbens (5)?
- Disambiguation: The *domain* is monetization, the *action* is execution
- Content: `stripe`, `subscription`, `payment_intent`, `invoice`
- **Result: 5 Nucleus Accumbens** (monetization domain wins because the webhook's purpose is billing)

### Example 5: "brain_classify MCP Tool"
- FUNKTION: Generates classification (language/AI tool)
- Content: `mcp`, `tool`, `classify`, `brain_mcp`, `region`
- **Result: 6 Broca-Areal** (AI/MCP tool for language-like classification)
