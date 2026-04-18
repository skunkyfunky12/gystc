"""Auto-classify notes into brain regions based on path, title + content keywords."""

from __future__ import annotations

import re

# Region index mapping:
# 0  Praefrontaler Cortex  — Architecture, decisions, planning
# 1  Motorischer Cortex    — API writes, actions, execution, code
# 2  Sensorischer Cortex   — Data intake, references, input, logs
# 3  Hippocampus           — Memory, sessions, personal notes
# 4  Kleinhirn             — Precision algorithms, routines, scripts
# 5  Nucleus Accumbens     — Subscriptions, pricing, rewards, goals
# 6  Broca-Areal           — AI, prompts, agents, language, LLM
# 7  Visueller Cortex      — UI, themes, design, CSS, frontend
# 8  Thalamus              — Index, MOC, data relay, routing, MCP
# 9  Stammhirn             — Config, infrastructure, system (fallback)
# 10 Basalganglien         — Pipelines, ETL, background, CI/CD
# 11 Amygdala              — Auth, team, social, security, alerts

# ---------------------------------------------------------------------------
# Phase 1: Vault folder → region (strongest signal, direct mapping)
# ---------------------------------------------------------------------------
_PATH_PREFIX_TO_REGION: list[tuple[str, int]] = [
    ("00 Index/", 8),              # Thalamus
    ("01 Lucas/Memory - User/", 3),        # Hippocampus
    ("01 Lucas/Memory - Feedback/", 3),    # Hippocampus
    ("01 Lucas/", 3),              # Hippocampus
    ("02 Projekte/Memory - Projects/", 0), # Praefrontaler Cortex
    ("02 Projekte/", 0),           # Praefrontaler Cortex (catch-all, after Graph skip)
    ("03 Agenten/", 6),            # Broca-Areal
    ("04 Sessions/", 3),           # Hippocampus
    ("05 Referenzen/", 2),         # Sensorischer Cortex
]

# Paths where folder-based classification is SKIPPED (use keywords instead).
# Graphify output contains code-level nodes that need per-note classification.
_PATH_SKIP_PREFIXES: list[str] = [
    "02 Projekte/Neural Brain Graph/",
    "02 Projekte/D2D-Scout Graph/",
]

# ---------------------------------------------------------------------------
# Phase 2: Title patterns for graphify code-nodes
# ---------------------------------------------------------------------------
_CODE_NODE_RE = re.compile(
    r"^(?:"
    r"\.__?[a-z]"              # .method() or ._private()
    r"|__init__\.py"           # __init__.py variants
    r"|test_[a-z]"             # test_something
    r"|_seed\(\)"              # _seed() variants
    r"|[a-z_]+\.(?:py|ts|js|mjs|tsx|jsx|css|json|toml|yaml|yml|d\.ts)$"  # filename.ext
    r")",
    re.IGNORECASE,
)

_REGION_KEYWORDS: list[tuple[int, list[str]]] = [
    (0, [
        "architektur", "architecture", "entscheidung", "decision", "planung", "planning",
        "roadmap", "strategie", "strategy", "design-doc", "rfc", "proposal", "scope",
        "milestone", "phase", "audit", "konzept", "blueprint",
        "projekt", "project", "spec", "requirement",
    ]),
    (1, [
        "commit", "deploy", "migration", "endpoint", "crud", "mutation",
        "execute", "action", "webhook", "trigger", "release",
        "push", "merge", "pull-request", "ship", "rollout",
        "supabase", "database", "postgres", "postgis", "sql",
        "fetch", "request", "response", "http", "rest", "graphql",
    ]),
    (2, [
        "referenz", "reference", "datenquelle", "datasource",
        "sensor", "log", "monitoring", "grafana", "metrics", "telemetry",
        "sentry", "tracking", "analytics", "posthog", "observability",
        "zensus", "census", "osm", "openstreetmap", "overpass", "geodaten",
        "daten", "data", "statistik", "report", "export",
    ]),
    (3, [
        "session", "memory", "erinnerung", "tagebuch", "journal", "personal",
        "lucas", "notiz", "gedaechtnis", "erfahrung", "lernen",
        "lesson", "feedback", "retrospektive", "retro",
        "brain vault", "remember", "recall",
    ]),
    (4, [
        "algorithmus", "algorithm", "routine", "automation",
        "regex", "parser", "compiler", "berechnung", "calculation", "formel",
        "precision", "batch", "scheduler",
        "scoring", "gewichtung", "weight", "distance", "heuristik",
    ]),
    (5, [
        "subscription", "abo", "pricing", "preis", "payment", "zahlung",
        "stripe", "billing", "tier", "revenue", "umsatz", "monetar",
        "freemium", "premium", "trial", "coupon", "discount",
    ]),
    (6, [
        "prompt", "llm", "gpt", "openai", "anthropic",
        "embedding", "vector", "rag", "langchain",
        "neural", "inference", "context-window", "broca",
        "mcp-server", "tool-use", "function-calling",
        "brain_mcp", "faiss", "sentence-transformer",
    ]),
    (7, [
        "design", "css", "tailwind", "frontend", "component",
        "layout", "theme", "dark-mode", "animation", "svg", "icon",
        "responsive", "mobile", "figma", "brutalist",
        "shadcn", "react", "jsx", "tsx", "next.js", "visual",
        "landing", "showcase", "bento", "grid",
    ]),
    (8, [
        "moc", "map-of-content", "relay",
        "thalamus", "hub", "zentral", "overview", "uebersicht",
        "navigation", "sitemap", "registry",
    ]),
    (9, [
        "config", "konfiguration", "infrastructure", "infra", "vps",
        "docker", "nginx", "ssl", "domain", "dns", "environment",
        "setup", "install", "systemd",
        "dotfile", "gitignore", "eslint", "prettier",
    ]),
    (10, [
        "pipeline", "etl", "ci/cd", "github-action",
        "queue", "worker", "job", "schedule", "hook", "pre-commit",
        "lint", "jest", "vitest", "e2e",
        "watcher", "indexer", "scanner",
    ]),
    (11, [
        "auth", "login", "jwt", "oauth", "password",
        "security", "sicherheit", "permission", "role", "team", "member",
        "invite", "account", "rbac", "rate-limit", "firewall",
        "vulnerability", "xss", "csrf", "injection",
        "rls", "row-level", "policy",
    ]),
]

# Pre-compile patterns for performance
_COMPILED_PATTERNS: list[tuple[int, re.Pattern]] = [
    (region_idx, re.compile(r"(?:^|\b|(?<=[\s_./]))(?:" + "|".join(re.escape(kw.strip()) for kw in keywords) + r")(?:\b|(?=[\s_./])|$)", re.IGNORECASE))
    for region_idx, keywords in _REGION_KEYWORDS
]


def classify_region(title: str, content: str, *, path: str | None = None) -> int:
    """Classify a note into a brain region based on path, title + content keywords.

    Returns the region index (0-11). Falls back to 9 (Stammhirn) if no clear match.

    Priority: path prefix (if unambiguous) > keyword scoring.
    Title matches count 3x, content matches 1x.
    """
    # Phase 1: Path-based classification (strongest signal)
    if path:
        path_norm = path.replace("\\", "/")
        # Skip path-based classification for graphify output folders
        skip = any(path_norm.startswith(p) for p in _PATH_SKIP_PREFIXES)
        if not skip:
            for prefix, region_idx in _PATH_PREFIX_TO_REGION:
                if path_norm.startswith(prefix):
                    return region_idx

    # Phase 2: Code-node detection — graphify artifacts stay in their
    # natural region. If keywords match, use that; otherwise Stammhirn
    # is fine for raw code nodes.
    is_code_node = bool(_CODE_NODE_RE.match(title))

    # Phase 3: Keyword scoring
    scores = [0.0] * 12
    title_lower = title.lower()
    # Scan more content for better classification (1500 chars)
    content_sample = content[:1500].lower()

    for region_idx, pattern in _COMPILED_PATTERNS:
        title_hits = len(pattern.findall(title_lower))
        content_hits = len(pattern.findall(content_sample))
        scores[region_idx] += title_hits * 3.0 + content_hits * 1.0

    best = max(range(12), key=lambda i: scores[i])

    # Lower threshold for code nodes (they have short titles)
    threshold = 1.0 if is_code_node else 2.0
    if scores[best] >= threshold:
        return best

    return 9  # Stammhirn fallback
