"""Auto-classify notes into brain regions based on path, title + content keywords.

Implements the FUNKTION > FORM > THEMA decision tree with weighted keyword tiers
and disambiguation rules. See classification_spec.md for the full spec.
"""

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
# Layer 0: Explicit #brain/ tag detection
# ---------------------------------------------------------------------------
_BRAIN_TAG_RE = re.compile(r"#brain/([\w-]+)")

_BRAIN_SLUG_TO_IDX: dict[str, int] = {
    "praefrontaler-cortex": 0,
    "motorischer-cortex": 1,
    "sensorischer-cortex": 2,
    "hippocampus": 3,
    "kleinhirn": 4,
    "nucleus-accumbens": 5,
    "broca-areal": 6,
    "visueller-cortex": 7,
    "thalamus": 8,
    "stammhirn": 9,
    "basalganglien": 10,
    "amygdala": 11,
}

# ---------------------------------------------------------------------------
# Layer 1: Vault folder → region (strongest structural signal)
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
    "02 Projekte/Neural Brain/graphify/",
    "02 Projekte/D2D-Scout/graphify/",
]

# ---------------------------------------------------------------------------
# Title patterns for graphify code-nodes
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

# ---------------------------------------------------------------------------
# Layer 2: Weighted keyword tiers per region
#   Tier 1 (5x) = strong, unambiguous signal for this region
#   Tier 2 (2x) = moderate signal, could appear in related regions
#   Tier 3 (1x) = weak contextual signal
# ---------------------------------------------------------------------------
_REGION_KEYWORDS_WEIGHTED: list[tuple[int, list[tuple[str, int]]]] = [
    (0, [
        # Tier 1: strong planning/architecture signals
        ("architektur", 5), ("architecture", 5), ("entscheidung", 5), ("decision", 5),
        ("roadmap", 5), ("strategie", 5), ("strategy", 5), ("design-doc", 5),
        ("rfc", 5), ("proposal", 5), ("blueprint", 5), ("adr", 5),
        ("trade-off", 5), ("tradeoff", 5),
        # Tier 2: moderate
        ("planung", 2), ("planning", 2), ("scope", 2), ("milestone", 2),
        ("phase", 2), ("audit", 2), ("konzept", 2), ("spec", 2),
        ("requirement", 2), ("initiative", 2), ("epic", 2), ("objective", 2),
        ("prioritization", 2), ("evaluation", 2),
        # Tier 3: weak context
        ("projekt", 1), ("project", 1), ("goal", 1), ("vision", 1),
        ("review", 1),
    ]),
    (1, [
        # Tier 1: execution/deploy/write
        ("deploy", 5), ("migration", 5), ("endpoint", 5), ("crud", 5),
        ("mutation", 5), ("webhook", 5), ("rollout", 5), ("ship", 5),
        ("seed", 5), ("upsert", 5), ("insert", 5),
        # Tier 2: moderate
        ("commit", 2), ("execute", 2), ("action", 2), ("trigger", 2),
        ("release", 2), ("push", 2), ("merge", 2), ("pull-request", 2),
        ("supabase", 2), ("database", 2), ("postgres", 2), ("postgis", 2),
        ("sql", 2), ("graphql", 2), ("api-route", 2), ("server-action", 2),
        # Tier 3: weak
        ("fetch", 1), ("request", 1), ("response", 1), ("http", 1),
        ("rest", 1), ("handler", 1), ("controller", 1),
    ]),
    (2, [
        # Tier 1: monitoring/data intake
        ("monitoring", 5), ("grafana", 5), ("telemetry", 5), ("sentry", 5),
        ("observability", 5), ("posthog", 5), ("metrics", 5),
        ("datasource", 5), ("datenquelle", 5),
        # Tier 2: moderate
        ("referenz", 2), ("reference", 2), ("sensor", 2), ("log", 2),
        ("tracking", 2), ("analytics", 2), ("zensus", 2), ("census", 2),
        ("osm", 2), ("openstreetmap", 2), ("overpass", 2), ("geodaten", 2),
        ("statistik", 2), ("dashboard", 2), ("alert", 2), ("incident", 2),
        ("trace", 2), ("span", 2),
        # Tier 3: weak
        ("daten", 1), ("data", 1), ("report", 1), ("export", 1),
        ("import", 1), ("input", 1),
    ]),
    (3, [
        # Tier 1: memory/personal
        ("session", 5), ("erinnerung", 5), ("tagebuch", 5), ("journal", 5),
        ("gedaechtnis", 5), ("brain vault", 5), ("remember", 5), ("recall", 5),
        ("retrospektive", 5), ("retro", 5),
        # Tier 2: moderate
        ("memory", 2), ("personal", 2), ("lucas", 2), ("notiz", 2),
        ("erfahrung", 2), ("lernen", 2), ("lesson", 2), ("feedback", 2),
        ("diary", 2), ("reflection", 2), ("note-to-self", 2),
        # Tier 3: weak
        ("history", 1), ("past", 1), ("yesterday", 1),
    ]),
    (4, [
        # Tier 1: precision computation
        ("algorithmus", 5), ("algorithm", 5), ("berechnung", 5), ("calculation", 5),
        ("formel", 5), ("formula", 5), ("heuristik", 5), ("heuristic", 5),
        ("scoring", 5), ("gewichtung", 5),
        # Tier 2: moderate
        ("regex", 2), ("parser", 2), ("compiler", 2), ("precision", 2),
        ("distance", 2), ("interpolation", 2), ("normalization", 2),
        ("sigmoid", 2), ("decay", 2), ("threshold", 2), ("percentile", 2),
        ("median", 2), ("mean", 2), ("variance", 2), ("coefficient", 2),
        # Tier 3: weak
        ("routine", 1), ("automation", 1), ("weight", 1), ("score", 1),
        ("math", 1), ("compute", 1),
    ]),
    (5, [
        # Tier 1: monetization/pricing
        ("subscription", 5), ("abo", 5), ("pricing", 5), ("preis", 5),
        ("payment", 5), ("zahlung", 5), ("stripe", 5), ("billing", 5),
        ("revenue", 5), ("umsatz", 5), ("monetar", 5), ("invoice", 5),
        # Tier 2: moderate
        ("tier", 2), ("freemium", 2), ("premium", 2), ("trial", 2),
        ("coupon", 2), ("discount", 2), ("checkout", 2), ("paywall", 2),
        ("mrr", 2), ("churn", 2), ("ltv", 2), ("arpu", 2),
        # Tier 3: weak
        ("plan", 1), ("upgrade", 1), ("downgrade", 1),
    ]),
    (6, [
        # Tier 1: AI/language generation
        ("prompt", 5), ("llm", 5), ("anthropic", 5), ("embedding", 5),
        ("rag", 5), ("langchain", 5), ("inference", 5), ("context-window", 5),
        ("broca", 5), ("mcp-server", 5), ("tool-use", 5), ("function-calling", 5),
        ("brain_mcp", 5), ("faiss", 5), ("sentence-transformer", 5),
        ("agent", 5), ("chain-of-thought", 5), ("system-prompt", 5),
        # Tier 2: moderate
        ("gpt", 2), ("openai", 2), ("vector", 2), ("neural", 2),
        ("token", 2), ("completion", 2), ("fine-tune", 2), ("classifier", 2),
        ("nlp", 2), ("transformer", 2), ("model", 2), ("generation", 2),
        # Tier 3: weak
        ("ai", 1), ("ml", 1), ("training", 1),
    ]),
    (7, [
        # Tier 1: visual/UI
        ("css", 5), ("tailwind", 5), ("frontend", 5), ("component", 5),
        ("theme", 5), ("dark-mode", 5), ("brutalist", 5), ("shadcn", 5),
        ("figma", 5), ("responsive", 5), ("bento", 5),
        # Tier 2: moderate
        ("design", 2), ("layout", 2), ("animation", 2), ("svg", 2),
        ("icon", 2), ("mobile", 2), ("react", 2), ("jsx", 2), ("tsx", 2),
        ("next.js", 2), ("visual", 2), ("landing", 2), ("showcase", 2),
        ("grid", 2), ("color", 2), ("palette", 2), ("font", 2),
        ("typography", 2), ("gradient", 2), ("border", 2), ("shadow", 2),
        # Tier 3: weak
        ("ui", 1), ("ux", 1), ("style", 1), ("render", 1),
    ]),
    (8, [
        # Tier 1: index/routing/relay
        ("moc", 5), ("map-of-content", 5), ("thalamus", 5), ("registry", 5),
        ("sitemap", 5),
        # Tier 2: moderate
        ("relay", 2), ("hub", 2), ("zentral", 2), ("overview", 2),
        ("uebersicht", 2), ("navigation", 2), ("index", 2), ("catalog", 2),
        ("directory", 2), ("table-of-contents", 2),
        # Tier 3: weak
        ("link", 1), ("route", 1),
    ]),
    (9, [
        # Tier 1: config/infrastructure
        ("docker", 5), ("nginx", 5), ("systemd", 5), ("dockerfile", 5),
        ("docker-compose", 5), ("dotfile", 5),
        # Tier 2: moderate
        ("config", 2), ("konfiguration", 2), ("infrastructure", 2), ("infra", 2),
        ("vps", 2), ("ssl", 2), ("domain", 2), ("dns", 2), ("environment", 2),
        ("setup", 2), ("install", 2), ("gitignore", 2), ("eslint", 2),
        ("prettier", 2), ("env", 2), ("terraform", 2), ("ansible", 2),
        ("pm2", 2), ("supervisor", 2),
        # Tier 3: weak
        ("server", 1), ("host", 1), ("port", 1), ("volume", 1),
    ]),
    (10, [
        # Tier 1: orchestration/pipelines
        ("pipeline", 5), ("etl", 5), ("ci/cd", 5), ("github-action", 5),
        ("workflow", 5), ("cron", 5), ("cronjob", 5),
        # Tier 2: moderate
        ("queue", 2), ("worker", 2), ("job", 2), ("schedule", 2),
        ("hook", 2), ("pre-commit", 2), ("lint", 2), ("jest", 2),
        ("vitest", 2), ("e2e", 2), ("watcher", 2), ("indexer", 2),
        ("scanner", 2), ("orchestrat", 2), ("dag", 2), ("step", 2),
        ("build", 2), ("test-suite", 2), ("playwright", 2),
        # Tier 3: weak
        ("run", 1), ("task", 1), ("batch", 1),
    ]),
    (11, [
        # Tier 1: auth/security
        ("auth", 5), ("jwt", 5), ("oauth", 5), ("rbac", 5),
        ("rls", 5), ("row-level", 5), ("firewall", 5),
        ("vulnerability", 5), ("xss", 5), ("csrf", 5), ("injection", 5),
        # Tier 2: moderate
        ("login", 2), ("password", 2), ("security", 2), ("sicherheit", 2),
        ("permission", 2), ("role", 2), ("team", 2), ("member", 2),
        ("invite", 2), ("account", 2), ("rate-limit", 2), ("policy", 2),
        ("encryption", 2), ("certificate", 2), ("cors", 2), ("csp", 2),
        ("session-token", 2), ("api-key", 2), ("secret", 2),
        # Tier 3: weak
        ("access", 1), ("guard", 1), ("protect", 1),
    ]),
]

# ---------------------------------------------------------------------------
# Disambiguation rules: negative keywords to resolve conflicts
# ---------------------------------------------------------------------------
# When two regions tie or are close, these patterns break the tie.
_DISAMBIGUATION: list[tuple[frozenset[int], str, int]] = [
    # Kleinhirn (4) vs Basalganglien (10): orchestration keywords → 10
    (frozenset({4, 10}), r"pipeline|ci/cd|ci|schedule|queue|worker|cron|watcher|orchestrat|dag|workflow", 10),
    # Kleinhirn (4) vs Basalganglien (10): computation keywords → 4
    (frozenset({4, 10}), r"algorithm|formula|scoring|calculation|heuristic|berechnung|formel|sigmoid|decay", 4),
    # Motorischer Cortex (1) vs Basalganglien (10): scheduling → 10
    (frozenset({1, 10}), r"pipeline|ci/cd|github-action|workflow|cron|schedule|orchestrat", 10),
    # Sensorischer Cortex (2) vs Thalamus (8): routing/indexing → 8
    (frozenset({2, 8}), r"moc|map-of-content|registry|sitemap|index|table-of-contents|catalog", 8),
    # Broca-Areal (6) vs Praefrontaler Cortex (0): planning → 0
    (frozenset({6, 0}), r"roadmap|architecture|architektur|decision|entscheidung|rfc|proposal|blueprint", 0),
    # Stammhirn (9) vs Motorischer Cortex (1): execution → 1
    (frozenset({9, 1}), r"deploy|migration|endpoint|rollout|ship|seed|upsert|crud", 1),
    # Visueller Cortex (7) vs Motorischer Cortex (1): visual → 7
    (frozenset({7, 1}), r"css|tailwind|theme|dark-mode|brutalist|figma|shadcn|animation|svg|layout|responsive", 7),
]

_DISAMBIGUATION_COMPILED: list[tuple[frozenset[int], re.Pattern, int]] = [
    (regions, re.compile(pattern, re.IGNORECASE), winner)
    for regions, pattern, winner in _DISAMBIGUATION
]

# ---------------------------------------------------------------------------
# Compile flat keyword patterns for scoring
# ---------------------------------------------------------------------------
_COMPILED_WEIGHTED: list[tuple[int, list[tuple[re.Pattern, int]]]] = []
for _region_idx, _kw_list in _REGION_KEYWORDS_WEIGHTED:
    _patterns = []
    for _kw, _weight in _kw_list:
        _pat = re.compile(
            r"(?:^|\b|(?<=[\s_./]))" + re.escape(_kw.strip()) + r"(?:\b|(?=[\s_./])|$)",
            re.IGNORECASE,
        )
        _patterns.append((_pat, _weight))
    _COMPILED_WEIGHTED.append((_region_idx, _patterns))


def classify_region(title: str, content: str, *, path: str | None = None) -> int:
    """Classify a note into a brain region.

    Priority chain:
      1. Explicit #brain/<slug> tag → instant return
      2. Path prefix mapping (structural)
      3. Weighted keyword scoring + disambiguation

    Returns the region index (0-11). Falls back to 9 (Stammhirn).
    """
    # --- Layer 0: Explicit #brain/ tag (highest priority) ---
    # Scan full content — tags can be at start (frontmatter) or end (appended)
    tag_match = _BRAIN_TAG_RE.search(content) if content else None
    if not tag_match:
        tag_match = _BRAIN_TAG_RE.search(title)
    if tag_match:
        slug = tag_match.group(1)
        idx = _BRAIN_SLUG_TO_IDX.get(slug)
        if idx is not None:
            return idx

    # --- Layer 1: Path-based classification ---
    if path:
        path_norm = path.replace("\\", "/")
        skip = any(path_norm.startswith(p) for p in _PATH_SKIP_PREFIXES)
        if not skip:
            for prefix, region_idx in _PATH_PREFIX_TO_REGION:
                if path_norm.startswith(prefix):
                    return region_idx

    # --- Code-node detection ---
    is_code_node = bool(_CODE_NODE_RE.match(title))

    # --- Layer 2: Weighted keyword scoring ---
    scores = [0.0] * 12
    title_lower = title.lower()
    content_sample = content[:3000].lower()  # increased from 1500

    for region_idx, patterns in _COMPILED_WEIGHTED:
        for pat, weight in patterns:
            title_hits = len(pat.findall(title_lower))
            content_hits = len(pat.findall(content_sample))
            # Title matches count 3x base, content 1x base, multiplied by tier weight
            scores[region_idx] += title_hits * 3.0 * weight + content_hits * 1.0 * weight

    # Find top two candidates
    ranked = sorted(range(12), key=lambda i: scores[i], reverse=True)
    best = ranked[0]
    second = ranked[1]

    # --- Disambiguation ---
    if scores[best] > 0 and scores[second] > 0:
        ratio = scores[second] / scores[best] if scores[best] > 0 else 0
        if ratio > 0.5:  # close contest
            candidates = frozenset({best, second})
            disambig_text = title_lower + " " + content_sample[:1500]
            for regions, pattern, winner in _DISAMBIGUATION_COMPILED:
                if candidates == regions:
                    if pattern.search(disambig_text):
                        best = winner
                        break

    # Apply threshold
    threshold = 1.0 if is_code_node else 2.0
    if scores[best] >= threshold:
        return best

    return 9  # Stammhirn fallback
