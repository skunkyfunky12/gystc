// Bridge: accept real graph data from Python host (set by QWebEngineScript before module load)
window.__graphData = window.__graphData || null;
window.__onNodeClick = window.__onNodeClick || null;

// Neural Brain — interactive prototype
// Synaptic + bioluminescent rendering using Three.js

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

/* ========================================================================
   DATA — 12 regions × cold palette (blues/teals/purples)
   ======================================================================== */

// Region anchors — spread into a larger brain-shaped volume so clusters breathe.
// Deep/central regions (thalamus, basal, nucleus, amygdala) stay near core;
// cortical regions pushed further out along their lobes.
const REGIONS = [
  { key: 'prefrontal', name: 'Präfrontaler Cortex',  subtitle: 'Entscheidungen · Planung', color: '#4DA3FF', pos: [   0, 420, -540] },
  { key: 'motor',      name: 'Motorischer Cortex',   subtitle: 'Code · Commits · Actions', color: '#22C55E', pos: [-210, 470, -260] },
  { key: 'sensory',    name: 'Sensorischer Cortex',  subtitle: 'Dateien · Inputs · Logs', color: '#5EE9F0', pos: [ 210, 470, -130] },
  { key: 'hippo',      name: 'Hippocampus',          subtitle: 'Langzeitgedächtnis · Projekte', color: '#9D7CFF', pos: [-340,   0,  130] },
  { key: 'cerebellum', name: 'Kleinhirn',            subtitle: 'Routinen · Skripte · Workflows', color: '#EC4899', pos: [ 280,-150,  470] },
  { key: 'nucleus',    name: 'Nucleus Accumbens',    subtitle: 'Ziele · Prioritäten · Wins', color: '#FACC15', pos: [  75, 225, -300] },
  { key: 'broca',      name: 'Broca-Areal',          subtitle: 'Sprache · Prompts · Schreiben', color: '#F97316', pos: [-260, 210, -340] },
  { key: 'visual',     name: 'Visueller Cortex',     subtitle: 'Design · UI · Diagramme', color: '#A855F7', pos: [ 210, 130,  470] },
  { key: 'thalamus',   name: 'Thalamus',             subtitle: 'Context-Router · MCP · API', color: '#3FD4E8', pos: [   0, 260,   75] },
  { key: 'stem',       name: 'Stammhirn',            subtitle: 'System · Config · Infra', color: '#94A3B8', pos: [   0,-420,  340] },
  { key: 'basal',      name: 'Basalganglien',        subtitle: 'Habits · Tools · Patterns', color: '#F43F5E', pos: [-225, 150, -120] },
  { key: 'amygdala',   name: 'Amygdala',             subtitle: 'Alerts · Fehler · Risiken', color: '#FB923C', pos: [-150, -95, -210] },
];

const PALETTES = {
  cold:   ['#3B82F6','#06B6D4','#10B981','#8B5CF6','#EC4899','#F59E0B','#EF4444','#84CC16','#14B8A6','#A855F7','#F97316','#6366F1'],
  teal:   ['#14B8A6','#3B82F6','#10B981','#FACC15','#F97316','#EC4899','#A855F7','#06B6D4','#84CC16','#EF4444','#8B5CF6','#F59E0B'],
  violet: ['#A855F7','#EC4899','#3B82F6','#F59E0B','#10B981','#EF4444','#06B6D4','#FACC15','#8B5CF6','#84CC16','#F97316','#14B8A6'],
  warm:   ['#F97316','#E11D48','#FACC15','#10B981','#3B82F6','#A855F7','#EC4899','#F59E0B','#06B6D4','#84CC16','#8B5CF6','#EF4444'],
  mono:   ['#5EE9F0','#4AD0B8','#B07CFF','#F0A055','#7C9AFF','#F07D5E','#4FA9C9','#FFD166','#EC4899','#22C55E','#9D7CFF','#FF7C7C'],
  // Graphite — grayscale with subtle cool undertones so regions stay distinguishable
  graphite: ['#E8ECF2','#C9D1DB','#A8B1BF','#8B94A3','#6F7889','#5A6374','#484F5E','#3C424F','#D4DAE3','#9AA3B2','#757E8E','#5F6878'],
  // Noir — deeper, warmer-to-cool grays for a moodier look
  noir:   ['#EFEFEF','#CFCFCF','#A9A9A9','#8A8A8A','#6E6E6E','#575757','#474747','#3A3A3A','#DCDCDC','#9A9A9A','#7A7A7A','#626262'],
};

// 12 communities per region, spread deterministically
const COMMUNITY_TO_REGION = [0,1,2,3,4,5,6,7, 7,7,10,10,8,8,10,11, 5,10,10,11,7,11,9,8, 10,10,10,11,7,3,3,3, 9,0,0,11,11,9,7,7, 9,7,7,9,3,3,3,7, 10,5,7,9,9,9,9,9, 9,9,9,9,9,0,9,8, 9,9,9,0,8,7,3,7, 7,3,0,10,8];

/* ========================================================================
   GENERATE GRAPH — realistic fake Obsidian vault
   ======================================================================== */

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function seededRandom(seed) {
  let s = seed | 0;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return ((s >>> 0) % 1_000_000) / 1_000_000;
  };
}

const rng = seededRandom(42);

const NOTE_TITLES = {
  prefrontal: ['Roadmap Q2 2026','Decision: Migration zu pnpm','Architektur Neural-Brain','Projekt-Priorisierung','Design-Review Notes','Tech-Debt Backlog','Feature-Spec: Graph-Layer','Planning: Refactor GL-Widget','Strategie: Claude-Memory','OKRs persönlich','Trade-off Analyse','Entscheidung: Monorepo'],
  motor: ['Commit e4f9a1 — physics tick','PR #142: Edge-Shader Fix','Refactor scene.py','Neue Funktion: pickNode','Bug-Fix: Memory-Leak OrbitCamera','Deploy script v3','Automation: test-runner','CI/CD Pipeline'],
  sensory: ['Graphify Output 2026-04-17','Log-Dump OrbitCamera','Screenshot: Brain-View','Input: PRD von Lisa','Dateibaum neural-brain/','Diff gegen main','Fehler-Log launch.txt'],
  hippo: ['Projekt: Neural-Brain','Projekt: Graphify','Projekt: Dashboard-Redesign','Notes 2024 retrospektive','Interview mit Max','Onboarding Woche 1','Conf-Talk PyCon 2025','Vault-Migration','Meeting mit Anna','Learning: OpenGL Shader','Research: Graph-Algos','Memory-Konsolidierung'],
  cerebellum: ['Daily Standup Template','Morning Routine','Weekly Review Flow','Pomodoro Ritual','Git-Workflow','PR-Review Checklist'],
  nucleus: ['Ziel: Prototype bis 30.04','Win: GL-Widget läuft','Reward: Side-Project shipped','Priorität A: Memory-Layer','Motivation: Demo-Video','Dopamin-Loop Insight'],
  broca: ['Prompt-Library Claude','System-Prompt v4','README Neural-Brain','Blog-Entwurf','Pitch-Deck Slides','Cold-Email Template'],
  visual: ['Figma Mockup Dashboard','Brand Colors 2026','Logo-Exploration','Icon-Set','Brain Visual Style','Diagramm Architektur','Moodboard dark-mode'],
  thalamus: ['MCP-Endpoint Liste','API-Keys vault','Context-Budget Rechner','Routing-Rules Claude','Integration Obsidian','Notion-Connector','Graphify Schema'],
  stem: ['dotfiles','~/.config/nvim','Docker-compose prod','Backup-Strategy','System-Setup MacBook','Environment Variables','Launch-Script'],
  basal: ['Tool: ripgrep Cheats','Habit: daily notes','Pattern: Error-Boundary','Snippet-Library','Alias-Liste','Hotkey-Setup','Raycast-Workflows'],
  amygdala: ['Alert: API-Key exposed','Risk: Obsidian-Plugin broken','Fehler: Physics NaN','Post-mortem Crash','Security-Audit','TODO: Kritischer Bug','Warnung: Disk 95%'],
};

// Pad each region up to ~42 titles with contextual variations (total ~500 nodes)
const TITLE_SUFFIXES = {
  prefrontal: ['2026-W12','2026-W13','2026-W14','Entwurf','final','v2','v3','Review','Offene Fragen','Follow-up','Rev A','Rev B','Outline','Draft','Sync','Status','Update','Rückblick','Ausblick','gesplittet','archiviert'],
  motor:      ['hotfix','WIP','rebase','cherry-pick','squash','refactor-pass','clean-up','lint-fix','perf','bundle-opt','tree-shake','v1.2','v1.3','v2.0','chore','fix/ui','fix/api','feat/','chore/deps','fix: race','RFC'],
  sensory:    ['2026-04-12','2026-04-14','2026-04-15','2026-04-16','chunk-07','chunk-08','batch-A','batch-B','stream-01','stream-02','trace-001','trace-002','snapshot','heap-dump','coredump','ingest-3','ingest-4','parse-log','diff-log','stderr','stdout'],
  hippo:      ['Dezember 2024','Januar 2025','Februar 2025','März 2025','Sprint 7','Sprint 8','Sprint 9','Retro Q1','Retro Q2','Interview mit Jonas','Interview mit Sarah','Meeting-Notes 04','Meeting-Notes 05','Lessons Learned','Knowledge-Dump','2024-Review','History','archiv-2023','archiv-2024','Milestone-Doc','Deep-Dive'],
  cerebellum: ['Morning','Evening','Monday','Friday','Sprint-Kickoff','Stand-up','Grooming','1:1 Template','Retro-Flow','Writing-Flow','Review-Flow','Debug-Ritual','Deploy-Flow','Release-Flow','Ship-It-Flow','Post-Ship','Cooldown','Energy-Check','Habit-Tracker','Streak-Day','Template'],
  nucleus:    ['30.04','15.05','01.06','Q2','Q3','Week 15','Week 16','Milestone 1','Milestone 2','Reward-Log','Win-Log Apr','Win-Log Mai','Streak 12','Streak 30','Focus-Day','Deep-Work','Big Goal','Small Win','Momentum','Energy-Log','Done-List'],
  broca:      ['Summarize','Extract','Classify','Rewrite','Expand','Simplify','Translate','Outline','Bullet-List','Q&A','Cold-Outreach','Landing-Copy','Subject-Line','Hook','TL;DR','Ghost-Draft','Interview-Q','Talking-Points','Script','Tweet-Thread','Newsletter'],
  visual:     ['Dashboard v2','Dashboard v3','Settings Screen','Onboarding Flow','Empty-State','Error-State','Loading-State','Mobile Layout','Desktop Layout','Dark Variant','Light Variant','Icon: search','Icon: brain','Icon: node','Poster','Social-Card','OG-Image','Logo v2','Color-Tokens','Spacing-Scale','Type-Scale'],
  thalamus:   ['OpenAI','Claude','Linear','Notion','Obsidian','GitHub','Slack','Calendar','Gmail','Drive','Firecrawl','Exa','Perplexity','Memory-Mcp','File-Mcp','Web-Mcp','Shell-Mcp','Ollama','Replicate','Supabase','Redis'],
  stem:       ['zshrc','bashrc','tmux.conf','gitconfig','ssh-config','brew-list','asdf-versions','pyproject','package.json','Dockerfile','.env.local','.env.prod','launchd','systemd','cronjob','ufw-rules','DNS','Backup-Script','Restore-Script','Monitoring','Healthcheck'],
  basal:      ['rg','fd','fzf','bat','eza','zoxide','starship','wezterm','kitty','neovim','helix','lazygit','delta','difftastic','jq','yq','mise','direnv','just','make','Raycast'],
  amygdala:   ['CVE-2024-XXXX','Leak: token','Leak: secret','Rate-Limit hit','OOM killed','Segfault','Timeout','5xx-Spike','4xx-Spike','Circuit-Open','Replay-Attack','Outdated dep','EOL warning','Disk-Full','CPU-Saturated','Memory-Pressure','Downtime 03:12','Incident #14','Post-mortem #14','Root-Cause','Mitigation'],
};
Object.keys(NOTE_TITLES).forEach(k => {
  const base = NOTE_TITLES[k];
  const sfx = TITLE_SUFFIXES[k] || [];
  const extra = [];
  for (let i = 0; i < sfx.length; i++) {
    const stem = base[i % base.length].split(' ')[0].replace(':', '');
    extra.push(`${stem} · ${sfx[i]}`);
  }
  NOTE_TITLES[k] = base.concat(extra);
});

function buildGraph() {
  if (window.__graphData) {
    return buildGraphFromData(window.__graphData);
  }
  const nodes = [];
  const nodesByRegion = Object.fromEntries(REGIONS.map(r => [r.key, []]));
  let id = 0;
  REGIONS.forEach((region, ri) => {
    const titles = NOTE_TITLES[region.key];
    for (let i = 0; i < titles.length; i++) {
      const title = titles[i];
      // Spherical scatter with gaussian radial falloff — tight core, soft round edge.
      // Sample a direction uniformly on the unit sphere, then scale by |N(0,1)|.
      const u = rng() * 2 - 1;
      const theta = rng() * Math.PI * 2;
      const s = Math.sqrt(1 - u * u);
      const dirX = Math.cos(theta) * s;
      const dirY = u;
      const dirZ = Math.sin(theta) * s;
      // approx |N(0, ~0.7)| via sum of uniforms → then abs; keeps core dense
      const radial = Math.abs((rng() + rng() + rng() + rng() - 2) * 0.9);
      const scatter = 8.5;
      const r = radial * scatter;
      const pos = new THREE.Vector3(
        region.pos[0] + dirX * r,
        region.pos[1] + dirY * r,
        region.pos[2] + dirZ * r,
      );
      const created = new Date(2025, Math.floor(rng() * 12), 1 + Math.floor(rng() * 27));
      const wordCount = 150 + Math.floor(rng() * 2500);
      const tags = pickTags(region.key);
      const node = { id: id++, title, region: region.key, regionIdx: ri, pos, created, wordCount, tags, degree: 0, hub: false };
      nodes.push(node);
      nodesByRegion[region.key].push(node);
    }
  });

  // Edges: 70% within-region, 30% cross-region
  const edges = [];
  const edgeSet = new Set();
  const addEdge = (a, b) => {
    if (a === b) return;
    const k = a < b ? a + ',' + b : b + ',' + a;
    if (edgeSet.has(k)) return;
    edgeSet.add(k);
    edges.push([a, b]);
    nodes[a].degree++;
    nodes[b].degree++;
  };

  REGIONS.forEach(r => {
    const rn = nodesByRegion[r.key];
    const density = 1.8;
    for (let i = 0; i < rn.length; i++) {
      const target = Math.max(1, Math.floor(density + rng() * 2));
      for (let t = 0; t < target; t++) {
        const j = Math.floor(rng() * rn.length);
        addEdge(rn[i].id, rn[j].id);
      }
    }
  });

  // Cross-region
  for (let i = 0; i < nodes.length; i++) {
    if (rng() < 0.28) {
      const other = Math.floor(rng() * nodes.length);
      if (nodes[other].region !== nodes[i].region) {
        addEdge(i, other);
      }
    }
  }

  // Mark hubs (max degree per region)
  REGIONS.forEach(r => {
    const rn = nodesByRegion[r.key];
    let best = rn[0];
    rn.forEach(n => { if (n.degree > best.degree) best = n; });
    if (best) best.hub = true;
  });

  // Organic relaxation: connected nodes pull together, overlapping nodes push apart.
  // A cheap force pass so clusters blend rather than sit as rigid blobs.
  relaxLayout(nodes, edges, 40);

  return { nodes, edges, nodesByRegion };
}

function buildGraphFromData(data) {
  const nodes = [];
  const nodesByRegion = Object.fromEntries(REGIONS.map(r => [r.key, []]));

  data.nodes.forEach((n, i) => {
    const regionIdx = n.regionIdx;
    const regionKey = REGIONS[regionIdx].key;
    const pos = new THREE.Vector3(n.pos[0], n.pos[1], n.pos[2]);
    const node = {
      id: i,
      title: n.title || n.id || `Node ${i}`,
      source_file: n.source_file || '',
      region: regionKey,
      regionIdx: regionIdx,
      pos: pos,
      created: new Date(n.created || Date.now()),
      wordCount: n.wordCount || 500,
      tags: n.tags || [],
      degree: 0,
      hub: false,
    };
    nodes.push(node);
    nodesByRegion[regionKey].push(node);
  });

  const edges = [];
  const edgeSet = new Set();
  const addEdge = (a, b) => {
    if (a === b) return;
    const k = a < b ? a + ',' + b : b + ',' + a;
    if (edgeSet.has(k)) return;
    edgeSet.add(k);
    edges.push([a, b]);
    nodes[a].degree++;
    nodes[b].degree++;
  };

  data.edges.forEach(e => addEdge(e[0], e[1]));

  // Mark hubs
  REGIONS.forEach(r => {
    const rn = nodesByRegion[r.key];
    if (!rn.length) return;
    let best = rn[0];
    rn.forEach(n => { if (n.degree > best.degree) best = n; });
    if (best) best.hub = true;
  });

  // Skip relaxLayout when using pre-computed positions from Python physics
  return { nodes, edges, nodesByRegion };
}

function relaxLayout(nodes, edges, iterations) {
  const tmp = nodes.map(() => new THREE.Vector3());
  const ideal = 3.2;          // desired edge length
  const repel = 14;           // node repulsion radius
  const center = new THREE.Vector3();
  const delta = new THREE.Vector3();
  for (let it = 0; it < iterations; it++) {
    tmp.forEach(v => v.set(0, 0, 0));

    // Attract along edges (springs)
    for (const [ai, bi] of edges) {
      const a = nodes[ai].pos, b = nodes[bi].pos;
      delta.subVectors(b, a);
      const len = delta.length() || 0.0001;
      const diff = (len - ideal) * 0.05;
      delta.multiplyScalar(diff / len);
      tmp[ai].add(delta);
      tmp[bi].sub(delta);
    }

    // Repel overlapping nodes (local, O(N·k) sampled)
    for (let i = 0; i < nodes.length; i++) {
      // sample 12 neighbours by index for efficiency
      for (let k = 1; k <= 12; k++) {
        const j = (i + k * 17) % nodes.length;
        if (j === i) continue;
        delta.subVectors(nodes[j].pos, nodes[i].pos);
        const d2 = delta.lengthSq();
        if (d2 < repel * repel && d2 > 0.01) {
          const push = (repel - Math.sqrt(d2)) * 0.06;
          delta.normalize().multiplyScalar(push);
          tmp[i].sub(delta);
          tmp[j].add(delta);
        }
      }
    }

    // Mild gravity back towards region anchor (keeps clusters coherent)
    for (let i = 0; i < nodes.length; i++) {
      const r = REGIONS[nodes[i].regionIdx];
      center.set(r.pos[0], r.pos[1], r.pos[2]);
      delta.subVectors(center, nodes[i].pos).multiplyScalar(0.02);
      tmp[i].add(delta);
    }

    for (let i = 0; i < nodes.length; i++) {
      nodes[i].pos.add(tmp[i].clampLength(0, 2));
    }
  }
}

function pickTags(regionKey) {
  const pool = {
    prefrontal: ['#decision','#plan','#roadmap','#strategy','#arch'],
    motor: ['#code','#commit','#pr','#deploy','#script'],
    sensory: ['#log','#input','#file','#screenshot','#diff'],
    hippo: ['#project','#memory','#notes','#history','#meeting'],
    cerebellum: ['#routine','#workflow','#ritual','#template'],
    nucleus: ['#goal','#win','#priority','#reward'],
    broca: ['#prompt','#writing','#doc','#readme'],
    visual: ['#design','#figma','#ui','#brand'],
    thalamus: ['#mcp','#api','#integration','#routing'],
    stem: ['#config','#infra','#system','#dotfiles'],
    basal: ['#tool','#habit','#snippet','#pattern'],
    amygdala: ['#alert','#bug','#risk','#security'],
  }[regionKey];
  const n = 1 + Math.floor(rng() * 3);
  const picked = new Set();
  while (picked.size < n) picked.add(pool[Math.floor(rng() * pool.length)]);
  return [...picked];
}

/* ========================================================================
   THREE.JS SCENE
   ======================================================================== */

const state = {
  palette: 'cold',
  glow: TWEAK_DEFAULTS.glow,
  stars: TWEAK_DEFAULTS.stars,
  nodeSize: TWEAK_DEFAULTS.nodeSize,
  pulseSpeed: TWEAK_DEFAULTS.pulseSpeed,
  bloom: TWEAK_DEFAULTS.bloom != null ? TWEAK_DEFAULTS.bloom : 0.55,
  bloomRadius: TWEAK_DEFAULTS.bloomRadius != null ? TWEAK_DEFAULTS.bloomRadius : 0.7,
  edgeOpacity: TWEAK_DEFAULTS.edgeOpacity || 0.45,
  edgeRange: TWEAK_DEFAULTS.edgeRange || 1.0,
  intraOnly: TWEAK_DEFAULTS.intraOnly || false,
  autoRotate: true,
  activeRegion: null,       // filtering
  hoverId: null,
  selectedId: null,
  heatmap: false,
};
state.palette = TWEAK_DEFAULTS.palette || 'cold';

const graph = buildGraph();
console.log(`Graph: ${graph.nodes.length} nodes, ${graph.edges.length} edges`);

const canvas = document.getElementById('three-canvas');
const canvasWrap = document.getElementById('canvas-wrap');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  preserveDrawingBuffer: true,
  alpha: false,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;

// Prevent blank frames on WebGL context loss (QWebEngine/Chromium compositing)
canvas.addEventListener('webglcontextlost', (e) => {
  e.preventDefault();
  console.warn('WebGL context lost — preventing default cleanup');
});
canvas.addEventListener('webglcontextrestored', () => {
  console.log('WebGL context restored');
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  onResize();
});

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070B);
scene.fog = new THREE.FogExp2(0x05070B, 0.002);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 4000);
camera.position.set(0, 100, 800);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.6;
controls.zoomSpeed = 0.8;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.12;
controls.minDistance = 50;
controls.maxDistance = 2000;

/* ---------- WASD FLY CONTROLS ---------- */

const _flyKeys = { w: false, a: false, s: false, d: false, q: false, e: false, shift: false };
const FLY_SPEED = 3.0;
const FLY_SPEED_FAST = 8.0;

document.addEventListener('keydown', (ev) => {
  const k = ev.key.toLowerCase();
  if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
  if (k in _flyKeys) _flyKeys[k] = true;
});
document.addEventListener('keyup', (ev) => {
  const k = ev.key.toLowerCase();
  if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
  if (k in _flyKeys) _flyKeys[k] = false;
});

function updateFlyControls() {
  const speed = _flyKeys.shift ? FLY_SPEED_FAST : FLY_SPEED;
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const up = new THREE.Vector3(0, 1, 0);

  const move = new THREE.Vector3();
  if (_flyKeys.w) move.add(forward);
  if (_flyKeys.s) move.sub(forward);
  if (_flyKeys.d) move.add(right);
  if (_flyKeys.a) move.sub(right);
  if (_flyKeys.e) move.add(up);
  if (_flyKeys.q) move.sub(up);

  if (move.lengthSq() > 0) {
    move.normalize().multiplyScalar(speed);
    camera.position.add(move);
    controls.target.add(move);
  }
}

/* ---------- STARS ---------- */

let starsObj = null;
function buildStars(count) {
  if (starsObj) {
    scene.remove(starsObj);
    starsObj.geometry.dispose();
    starsObj.material.dispose();
  }
  if (count <= 0) { starsObj = null; return; }
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);
  const brightness = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    // sphere shell
    const r = 900 + Math.random() * 900;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i*3]   = r * Math.sin(phi) * Math.cos(theta);
    positions[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i*3+2] = r * Math.cos(phi);
    brightness[i] = 0.2 + Math.random() * 0.8;
  }
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('bright', new THREE.BufferAttribute(brightness, 1));

  const mat = new THREE.ShaderMaterial({
    uniforms: { u_time: { value: 0 } },
    vertexShader: `
      attribute float bright;
      varying float v_bright;
      uniform float u_time;
      void main() {
        v_bright = bright;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        float twinkle = 0.85 + 0.15 * sin(u_time * 0.3 + bright * 20.0);
        gl_PointSize = (1.0 + bright * 1.4) * twinkle * (300.0 / -mvPosition.z);
      }
    `,
    fragmentShader: `
      varying float v_bright;
      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float d = length(uv);
        if (d > 0.5) discard;
        float falloff = pow(smoothstep(0.5, 0.0, d), 1.6);
        float a = falloff * v_bright * 0.75;
        vec3 c = mix(vec3(0.88, 0.91, 0.96), vec3(0.78, 0.85, 0.95), v_bright);
        gl_FragColor = vec4(c, a);
      }
    `,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });

  starsObj = new THREE.Points(geom, mat);
  starsObj.frustumCulled = false;
  scene.add(starsObj);
}
buildStars(state.stars);

/* ---------- NODES ---------- */

const nodeGeometry = new THREE.BufferGeometry();
const nodePositions = new Float32Array(graph.nodes.length * 3);
const nodeColors = new Float32Array(graph.nodes.length * 3);
const nodeSizes = new Float32Array(graph.nodes.length);
const nodeRegionIdx = new Float32Array(graph.nodes.length);
const nodeAlpha = new Float32Array(graph.nodes.length);

function applyPaletteToNodes() {
  const palette = PALETTES[state.palette];
  const c = new THREE.Color();
  graph.nodes.forEach((n, i) => {
    c.set(palette[n.regionIdx]);
    nodeColors[i*3]   = c.r;
    nodeColors[i*3+1] = c.g;
    nodeColors[i*3+2] = c.b;
  });
  nodeGeometry.attributes.color.needsUpdate = true;
}

const nodeActivation = new Float32Array(graph.nodes.length); // 0..1 decays over time (recency-glow)
const nodeConfidence = new Float32Array(graph.nodes.length); // 0..1 static-ish (# embeddings/backlinks)

graph.nodes.forEach((n, i) => {
  nodePositions[i*3]   = n.pos.x;
  nodePositions[i*3+1] = n.pos.y;
  nodePositions[i*3+2] = n.pos.z;
  nodeSizes[i] = n.hub ? 4.5 : (1.5 + Math.min(n.degree, 8) * 0.35);
  nodeRegionIdx[i] = n.regionIdx;
  nodeAlpha[i] = 1.0;
  // confidence = normalized degree + word-count factor
  nodeConfidence[i] = Math.min(1, n.degree / 14) * 0.7 + Math.min(1, n.wordCount / 2000) * 0.3;
  n.confidence = nodeConfidence[i];
  n.activation = 0;
  nodeActivation[i] = 0;
});

nodeGeometry.setAttribute('position', new THREE.BufferAttribute(nodePositions, 3));
nodeGeometry.setAttribute('color', new THREE.BufferAttribute(nodeColors, 3));
nodeGeometry.setAttribute('size', new THREE.BufferAttribute(nodeSizes, 1));
nodeGeometry.setAttribute('regionIdx', new THREE.BufferAttribute(nodeRegionIdx, 1));
nodeGeometry.setAttribute('alpha', new THREE.BufferAttribute(nodeAlpha, 1));
nodeGeometry.setAttribute('activation', new THREE.BufferAttribute(nodeActivation, 1));
nodeGeometry.setAttribute('confidence', new THREE.BufferAttribute(nodeConfidence, 1));
applyPaletteToNodes();

const nodeMaterial = new THREE.ShaderMaterial({
  uniforms: {
    u_time: { value: 0 },
    u_glow: { value: state.glow },
    u_sizeScale: { value: state.nodeSize },
    u_hoverId: { value: -1 },
    u_selectedId: { value: -1 },
    u_pixelRatio: { value: renderer.getPixelRatio() },
  },
  vertexShader: `
    attribute float size;
    attribute float regionIdx;
    attribute float alpha;
    attribute float activation;
    attribute float confidence;
    varying vec3 v_color;
    varying float v_alpha;
    varying float v_pulse;
    varying float v_act;
    varying float v_conf;
    uniform float u_time;
    uniform float u_sizeScale;
    uniform float u_hoverId;
    uniform float u_selectedId;
    uniform float u_pixelRatio;
    void main() {
      v_color = color;
      v_alpha = alpha;
      v_act = activation;
      v_conf = confidence;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;

      float id = float(gl_VertexID);
      // Dormant nodes: static. Active nodes: pulse proportional to activation.
      float pulse = 1.0 + activation * 0.3 * sin(u_time * 2.5 + id * 0.23);
      float s = size * u_sizeScale * pulse;
      s += activation * 5.0; // firing nodes swell visibly
      if (abs(id - u_hoverId) < 0.5) { s *= 2.4; pulse = 1.3; }
      if (abs(id - u_selectedId) < 0.5) { s *= 2.0; pulse = 1.6; }
      v_pulse = pulse + activation * 0.35;
      gl_PointSize = s * (400.0 / -mvPosition.z) * u_pixelRatio;
    }
  `,
  fragmentShader: `
    varying vec3 v_color;
    varying float v_alpha;
    varying float v_pulse;
    varying float v_act;
    varying float v_conf;
    uniform float u_glow;
    void main() {
      vec2 uv = gl_PointCoord - 0.5;
      float d = length(uv);
      if (d > 0.5) discard;

      // Bioluminescent core + halo + confidence-halo + activation-bloom
      float core = smoothstep(0.22, 0.08, d);
      float soma = smoothstep(0.12, 0.0, d);
      float halo = smoothstep(0.5, 0.38, d);
      float ang = atan(uv.y, uv.x);
      float rays = 0.5 + 0.5 * cos(ang * 6.0);
      float dendrite = rays * smoothstep(0.5, 0.18, d) * smoothstep(0.12, 0.25, d) * 0.22;
      float ring = smoothstep(0.42, 0.4, d) * smoothstep(0.35, 0.37, d);
      float confHalo = smoothstep(0.5, 0.4, d) * v_conf * 0.08;

      // Fresnel-Rim: thin bright edge for spherical appearance
      float rim = smoothstep(0.5, 0.46, d) * smoothstep(0.4, 0.46, d) * 0.55;

      // Smooth activation curve instead of linear — no harsh pixel flash
      float actSmooth = v_act * v_act * (3.0 - 2.0 * v_act);
      float actBloom = smoothstep(0.5, 0.05, d) * actSmooth * 0.95;
      float actCore  = smoothstep(0.3, 0.0, d)  * actSmooth * 1.3;

      vec3 col = v_color * (core * 1.4 + soma * 1.8 + halo * 0.1 * u_glow + dendrite + ring * 0.22 + confHalo + rim + actBloom + actCore);
      col += vec3(1.0, 1.0, 0.92) * (actCore * 0.4 + actBloom * 0.18 + rim * 0.15);
      float a = (core * 1.0 + soma * 0.7 + halo * 0.1 * u_glow + dendrite * 0.7 + confHalo * 0.2 + rim * 0.6 + actBloom * 0.85 + actCore * 0.7) * v_alpha * min(v_pulse, 1.15);
      gl_FragColor = vec4(min(col, 1.0), a);
    }
  `,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexColors: true,
});

const nodePoints = new THREE.Points(nodeGeometry, nodeMaterial);
scene.add(nodePoints);

/* ---------- HUB SPRITE RINGS (for hub nodes) ---------- */

const ringGroup = new THREE.Group();
graph.nodes.filter(n => n.hub).forEach(n => {
  const ringGeom = new THREE.RingGeometry(1.8, 2.0, 48);
  const ringMat = new THREE.MeshBasicMaterial({
    color: PALETTES[state.palette][n.regionIdx],
    transparent: true, opacity: 0.35, side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const ring = new THREE.Mesh(ringGeom, ringMat);
  ring.position.copy(n.pos);
  ring.userData.nodeId = n.id;
  ringGroup.add(ring);
});
scene.add(ringGroup);

/* ---------- PULSE RINGS (expanding ripple on neural firing) ---------- */
const pulseRings = [];
function spawnPulseRing(node) {
  const palette = PALETTES[state.palette];
  const color = new THREE.Color(palette[node.regionIdx]);
  const geom = new THREE.RingGeometry(0.5, 1.0, 64);
  const mat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.9,
    side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.copy(node.pos);
  mesh.lookAt(camera.position);
  scene.add(mesh);
  pulseRings.push({ mesh, t0: clock.elapsedTime, dur: 2.0, node });

  // Second ring delayed for double-pulse effect
  setTimeout(() => {
    const geom2 = new THREE.RingGeometry(0.5, 0.8, 64);
    const mat2 = new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: 0.6,
      side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const mesh2 = new THREE.Mesh(geom2, mat2);
    mesh2.position.copy(node.pos);
    mesh2.lookAt(camera.position);
    scene.add(mesh2);
    pulseRings.push({ mesh: mesh2, t0: clock.elapsedTime, dur: 1.8, node });
  }, 200);
}

function updatePulseRings(t) {
  for (let i = pulseRings.length - 1; i >= 0; i--) {
    const pr = pulseRings[i];
    const elapsed = t - pr.t0;
    const k = elapsed / pr.dur;
    if (k >= 1) {
      scene.remove(pr.mesh);
      pr.mesh.geometry.dispose();
      pr.mesh.material.dispose();
      pulseRings.splice(i, 1);
      continue;
    }
    // Expand ring outward, fade out
    const scale = 1 + k * 18;
    pr.mesh.scale.setScalar(scale);
    pr.mesh.material.opacity = (1 - k * k) * 0.7;
    pr.mesh.lookAt(camera.position);
  }
}

function applyPaletteToRings() {
  const palette = PALETTES[state.palette];
  ringGroup.children.forEach(ring => {
    const node = graph.nodes[ring.userData.nodeId];
    ring.material.color.set(palette[node.regionIdx]);
  });
}

/* ---------- EDGES (as LineSegments, per-edge color) ---------- */

const EDGE_SUBDIV = 10;
const edgeCount = graph.edges.length;
const segPerEdge = EDGE_SUBDIV;
const vertsPerEdge = segPerEdge * 2;
const edgePositions = new Float32Array(edgeCount * vertsPerEdge * 3);
const edgeColors = new Float32Array(edgeCount * vertsPerEdge * 3);
const edgeAlphas = new Float32Array(edgeCount * vertsPerEdge);
const edgeParams = new Float32Array(edgeCount * vertsPerEdge * 2);

const edgeCurveOffsets = new Float32Array(edgeCount * 3);
for (let i = 0; i < edgeCount; i++) {
  const dir = new THREE.Vector3(Math.random()-0.5, Math.random()-0.5, Math.random()-0.5).normalize();
  edgeCurveOffsets[i*3] = dir.x; edgeCurveOffsets[i*3+1] = dir.y; edgeCurveOffsets[i*3+2] = dir.z;
}

function updateEdgePositions() {
  const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), mid = new THREE.Vector3(), off = new THREE.Vector3();
  graph.edges.forEach((e, i) => {
    const a = graph.nodes[e[0]].pos;
    const b = graph.nodes[e[1]].pos;
    const dist = a.distanceTo(b);
    mid.addVectors(a, b).multiplyScalar(0.5);
    off.set(edgeCurveOffsets[i*3], edgeCurveOffsets[i*3+1], edgeCurveOffsets[i*3+2]).multiplyScalar(dist * 0.18);
    mid.add(off);
    for (let s = 0; s < segPerEdge; s++) {
      const t0 = s / segPerEdge;
      const t1 = (s + 1) / segPerEdge;
      const q = (t, out) => {
        const it = 1 - t;
        out.set(
          it*it*a.x + 2*it*t*mid.x + t*t*b.x,
          it*it*a.y + 2*it*t*mid.y + t*t*b.y,
          it*it*a.z + 2*it*t*mid.z + t*t*b.z
        );
      };
      q(t0, tmpA); q(t1, tmpB);
      const vi = i * vertsPerEdge + s * 2;
      edgePositions[vi*3]   = tmpA.x; edgePositions[vi*3+1] = tmpA.y; edgePositions[vi*3+2] = tmpA.z;
      edgePositions[vi*3+3] = tmpB.x; edgePositions[vi*3+4] = tmpB.y; edgePositions[vi*3+5] = tmpB.z;
      edgeParams[vi*2]     = t0; edgeParams[vi*2+1] = i;
      edgeParams[vi*2+2]   = t1; edgeParams[vi*2+3] = i;
    }
  });
}

function applyPaletteToEdges() {
  const palette = PALETTES[state.palette];
  const c1 = new THREE.Color(), c2 = new THREE.Color(), cm = new THREE.Color();
  graph.edges.forEach((e, i) => {
    const na = graph.nodes[e[0]], nb = graph.nodes[e[1]];
    c1.set(palette[na.regionIdx]); c2.set(palette[nb.regionIdx]);
    for (let s = 0; s < segPerEdge; s++) {
      const t0 = s / segPerEdge, t1 = (s + 1) / segPerEdge;
      cm.copy(c1).lerp(c2, t0);
      const vi = i * vertsPerEdge + s * 2;
      edgeColors[vi*3]   = cm.r; edgeColors[vi*3+1] = cm.g; edgeColors[vi*3+2] = cm.b;
      cm.copy(c1).lerp(c2, t1);
      edgeColors[vi*3+3] = cm.r; edgeColors[vi*3+4] = cm.g; edgeColors[vi*3+5] = cm.b;
    }
  });
  if (edgeGeom.attributes.color) edgeGeom.attributes.color.needsUpdate = true;
  if (edgeGeom.attributes.alpha) edgeGeom.attributes.alpha.needsUpdate = true;
}

const edgeGeom = new THREE.BufferGeometry();
updateEdgePositions();
edgeGeom.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3));
edgeGeom.setAttribute('color', new THREE.BufferAttribute(edgeColors, 3));
edgeGeom.setAttribute('alpha', new THREE.BufferAttribute(edgeAlphas, 1));
edgeGeom.setAttribute('params', new THREE.BufferAttribute(edgeParams, 2));
applyPaletteToEdges();

/* ---------- EDGE ACTIVATION SYSTEM (alpha-buffer based, no textures) ---------- */
const EDGE_CAP = Math.max(graph.edges.length, 1);
const edgeActivation = new Float32Array(EDGE_CAP);
const highlightedEdges = new Set();

function activateEdges(edgeIds) {
  for (const id of edgeIds) {
    if (id < EDGE_CAP) edgeActivation[id] = 1.0;
  }
  refreshEdgeAlphas();
}

function updateEdgeActivation(dt) {
  let anyActive = false;
  for (let i = 0; i < EDGE_CAP; i++) {
    if (edgeActivation[i] > 0) {
      edgeActivation[i] = Math.max(0, edgeActivation[i] - dt * 0.25);
      anyActive = true;
    }
  }
  if (anyActive) refreshEdgeAlphas();
}

function setHighlights(edgeIds) {
  highlightedEdges.clear();
  edgeIds.forEach(id => { if (id < EDGE_CAP) highlightedEdges.add(id); });
  refreshEdgeAlphas();
}

const tap = (t) => Math.pow(Math.sin(t * Math.PI), 0.6);

function refreshEdgeAlphas() {
  if (maxEdgeLength === 0) return; // not yet initialized
  const cutoff = state.edgeRange * maxEdgeLength;
  const opScale = state.edgeOpacity / 0.35;
  graph.edges.forEach(([a, b], i) => {
    const na = graph.nodes[a], nb = graph.nodes[b];
    const sameRegion = na.region === nb.region;
    const inRange = edgeLengths[i] <= cutoff;
    const show = inRange && (!state.intraOnly || sameRegion);
    const off = i * vertsPerEdge;
    if (!show) {
      for (let s = 0; s < segPerEdge; s++) {
        const vi = off + s * 2;
        edgeAlphas[vi] = 0.0; edgeAlphas[vi+1] = 0.0;
      }
      return;
    }
    const baseA = sameRegion ? 0.50 : 0.20;
    const act = edgeActivation[i] * 0.6;
    const hl = highlightedEdges.has(i) ? 0.8 : 0.0;
    for (let s = 0; s < segPerEdge; s++) {
      const t0 = s / segPerEdge, t1 = (s + 1) / segPerEdge;
      const vi = off + s * 2;
      edgeAlphas[vi]   = Math.min(1.0, baseA * tap(t0) * opScale + act + hl);
      edgeAlphas[vi+1] = Math.min(1.0, baseA * tap(t1) * opScale + act + hl);
    }
  });
  edgeGeom.attributes.alpha.needsUpdate = true;
}

const edgeMaterial = new THREE.ShaderMaterial({
  uniforms: {
    u_time: { value: 0 },
    u_pulseSpeed: { value: state.pulseSpeed },
  },
  vertexShader: `
    attribute float alpha;
    attribute vec2 params;
    varying vec3 v_color;
    varying float v_alpha;
    varying float v_t;
    varying float v_edgeId;
    void main() {
      v_color = color;
      v_alpha = alpha;
      v_t = params.x;
      v_edgeId = params.y;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    varying vec3 v_color;
    varying float v_alpha;
    varying float v_t;
    varying float v_edgeId;
    uniform float u_time;
    uniform float u_pulseSpeed;
    void main() {
      // Traveling pulse only on bright (activated/highlighted) edges
      float activation = smoothstep(0.75, 1.0, v_alpha);
      float phase = u_time * u_pulseSpeed * 0.55 + v_edgeId * 0.13;
      float pulse = smoothstep(0.12, 0.0, abs(fract(phase) - v_t)) * activation;
      float alpha = min(v_alpha + pulse * 0.5, 1.0);
      vec3 col = min(v_color * (1.0 + pulse * 1.2), 1.0);
      gl_FragColor = vec4(col, alpha);
    }
  `,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexColors: true,
});

const edgeLines = new THREE.LineSegments(edgeGeom, edgeMaterial);
scene.add(edgeLines);

// Build per-node edge index for highlighting neighbours
const nodeEdges = new Array(graph.nodes.length).fill(null).map(() => []);
graph.edges.forEach((e, i) => {
  nodeEdges[e[0]].push(i);
  nodeEdges[e[1]].push(i);
});

/* ---------- POSTPROCESSING (bloom for bioluminescence) ---------- */

// HalfFloat render target prevents black-block artifacts from additive blending overflow
const hdrRT = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
const composer = new EffectComposer(renderer, hdrRT);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.55, 0.7, 0.35);
bloomPass.strength = state.bloom;
bloomPass.radius = state.bloomRadius;
composer.addPass(bloomPass);
composer.addPass(new OutputPass());

/* ---------- RESIZE ---------- */

function onResize() {
  const w = canvasWrap.clientWidth, h = canvasWrap.clientHeight;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  bloomPass.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
onResize();
window.addEventListener('resize', onResize);

/* ========================================================================
   PICKING — since Points with custom shader don't raycast nicely,
   we do manual screen-space hit testing.
   ======================================================================== */

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 1.8;
const pointer = new THREE.Vector2();

function pickNode(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;

  // Manual screen-space picking for accuracy with sprite points
  const vec = new THREE.Vector3();
  let best = null, bestDist = Infinity;
  const w = rect.width, h = rect.height;
  for (let i = 0; i < graph.nodes.length; i++) {
    const n = graph.nodes[i];
    if (state.activeRegion && n.region !== state.activeRegion) continue;
    vec.copy(n.pos).project(camera);
    if (vec.z > 1 || vec.z < -1) continue;
    const sx = (vec.x * 0.5 + 0.5) * w;
    const sy = (-vec.y * 0.5 + 0.5) * h;
    const dx = sx - (clientX - rect.left);
    const dy = sy - (clientY - rect.top);
    const d2 = dx*dx + dy*dy;
    const size = (nodeSizes[i] * state.nodeSize) + 4;
    if (d2 < size*size && d2 < bestDist) { bestDist = d2; best = i; }
  }
  return best;
}

/* ========================================================================
   INTERACTIONS
   ======================================================================== */

const tooltip = document.getElementById('tooltip');
const hudNode = document.getElementById('hud-node');
const hudRegion = document.getElementById('hud-region');

canvas.addEventListener('mousemove', (e) => {
  const id = pickNode(e.clientX, e.clientY);
  state.hoverId = id;
  nodeMaterial.uniforms.u_hoverId.value = id ?? -1;
  if (id != null) {
    const n = graph.nodes[id];
    const regionName = REGIONS[n.regionIdx].name;
    const regionColor = PALETTES[state.palette][n.regionIdx];
    tooltip.classList.add('visible');
    tooltip.style.left = (e.clientX - canvas.getBoundingClientRect().left) + 'px';
    tooltip.style.top = (e.clientY - canvas.getBoundingClientRect().top) + 'px';
    tooltip.innerHTML = `
      <div class="t-title"><span class="t-dot" style="background:${regionColor};color:${regionColor}"></span>${esc(n.title)}</div>
      <div class="t-meta">GRAD ${n.degree} · ${n.wordCount} WÖRTER${n.hub ? ' · HUB' : ''}</div>
      <div class="t-region">${regionName}</div>
    `;
    hudNode.textContent = '#' + n.id.toString().padStart(4, '0');
    hudRegion.textContent = regionName.toUpperCase();
    canvas.style.cursor = 'pointer';

    // Highlight neighbour edges
    setHighlights(nodeEdges[id]);
  } else {
    tooltip.classList.remove('visible');
    hudNode.textContent = '—';
    hudRegion.textContent = '—';
    canvas.style.cursor = 'default';
    if (state.selectedId != null) setHighlights(nodeEdges[state.selectedId]);
    else setHighlights([]);
  }
});

canvas.addEventListener('mouseleave', () => {
  tooltip.classList.remove('visible');
  state.hoverId = null;
  nodeMaterial.uniforms.u_hoverId.value = -1;
});

let clickDownPos = null;
canvas.addEventListener('mousedown', (e) => { clickDownPos = [e.clientX, e.clientY]; });
canvas.addEventListener('mouseup', (e) => {
  if (!clickDownPos) return;
  const dx = e.clientX - clickDownPos[0], dy = e.clientY - clickDownPos[1];
  if (Math.sqrt(dx*dx + dy*dy) < 4) {
    const id = pickNode(e.clientX, e.clientY);
    if (id != null) selectNode(id);
    else clearSelection();
  }
  clickDownPos = null;
});

function selectNode(id) {
  state.selectedId = id;
  nodeMaterial.uniforms.u_selectedId.value = id;
  setHighlights(nodeEdges[id]);
  activateEdges(nodeEdges[id]);
  renderDetail(graph.nodes[id]);
  if (window.__onNodeClick) window.__onNodeClick(id, graph.nodes[id].title);
  // Focus camera on the node
  const n = graph.nodes[id];
  controls.target.lerp(n.pos, 0.35);
}
function clearSelection() {
  state.selectedId = null;
  nodeMaterial.uniforms.u_selectedId.value = -1;
  setHighlights([]);
  renderDetailEmpty();
}

/* ========================================================================
   UI WIRING
   ======================================================================== */

// Region list
const regionListEl = document.getElementById('region-list');
REGIONS.forEach((r, i) => {
  const count = graph.nodes.filter(n => n.region === r.key).length;
  const el = document.createElement('div');
  el.className = 'region';
  el.dataset.region = r.key;
  el.innerHTML = `
    <span class="dot" style="background:${PALETTES[state.palette][i]};color:${PALETTES[state.palette][i]}"></span>
    <span class="region-name">${r.name}<span class="region-sub">${r.subtitle || ''}</span></span>
    <span class="region-count">${count}</span>
  `;
  el.addEventListener('click', () => toggleRegion(r.key));
  regionListEl.appendChild(el);
});

function toggleRegion(key) {
  if (state.activeRegion === key) state.activeRegion = null;
  else state.activeRegion = key;

  document.querySelectorAll('.region').forEach(el => {
    el.classList.toggle('active', el.dataset.region === state.activeRegion);
    if (state.activeRegion) {
      el.classList.toggle('muted', el.dataset.region !== state.activeRegion);
    } else {
      el.classList.remove('muted');
    }
  });

  // Alpha mask
  for (let i = 0; i < graph.nodes.length; i++) {
    nodeAlpha[i] = (!state.activeRegion || graph.nodes[i].region === state.activeRegion) ? 1.0 : 0.06;
  }
  nodeGeometry.attributes.alpha.needsUpdate = true;

  // Edge alpha — must write ALL vertices per edge (vertsPerEdge), not just 2
  graph.edges.forEach((e, i) => {
    const na = graph.nodes[e[0]], nb = graph.nodes[e[1]];
    const active = !state.activeRegion || (na.region === state.activeRegion && nb.region === state.activeRegion);
    const base = na.region === nb.region ? 0.32 : 0.08;
    const a = active ? base : 0.01;
    const off = i * vertsPerEdge;
    for (let s = 0; s < vertsPerEdge; s++) edgeAlphas[off + s] = a;
  });
  edgeGeom.attributes.alpha.needsUpdate = true;

  updateStats();
}

// Palette swatches
const paletteOptions = [
  { key: 'cold',     label: 'kalt'     },
  { key: 'teal',     label: 'teal'     },
  { key: 'violet',   label: 'violet'   },
  { key: 'warm',     label: 'warm'     },
  { key: 'graphite', label: 'graphit'  },
  { key: 'noir',     label: 'noir'     },
];
const paletteEl = document.getElementById('palette-options');
paletteOptions.forEach(({ key, label }) => {
  const el = document.createElement('div');
  el.className = 'palette-opt' + (state.palette === key ? ' active' : '');
  el.dataset.palette = key;
  el.title = label;
  el.innerHTML = PALETTES[key].slice(0, 4).map(c => `<span style="background:${c}"></span>`).join('');
  el.addEventListener('click', () => setPalette(key));
  paletteEl.appendChild(el);
});

function setPalette(key) {
  state.palette = key;
  document.querySelectorAll('.palette-opt').forEach(el => el.classList.toggle('active', el.dataset.palette === key));
  document.getElementById('palette-name').textContent = paletteOptions.find(p => p.key === key).label;
  applyPaletteToNodes();
  applyPaletteToRings();
  applyPaletteToEdges();
  // Update sidebar dots
  document.querySelectorAll('.region').forEach((el, i) => {
    const dot = el.querySelector('.dot');
    dot.style.background = PALETTES[key][i];
    dot.style.color = PALETTES[key][i];
  });
  // Update floating 3D region-labels
  document.querySelectorAll('.region-label').forEach((el, i) => {
    const dot = el.querySelector('.rl-dot');
    if (dot) { dot.style.background = PALETTES[key][i]; dot.style.color = PALETTES[key][i]; }
  });
  persistTweaks({ palette: key });
}

// Sliders
function bindSlider(sliderId, valId, callback) {
  const s = document.getElementById(sliderId);
  const v = document.getElementById(valId);
  s.addEventListener('input', () => {
    const val = parseFloat(s.value);
    v.textContent = (sliderId === 'stars-slider') ? Math.round(val) : val.toFixed(2);
    callback(val);
  });
}
bindSlider('glow-slider', 'glow-val', (v) => {
  state.glow = v;
  nodeMaterial.uniforms.u_glow.value = v;
  persistTweaks({ glow: v });
});
bindSlider('stars-slider', 'stars-val', (v) => {
  state.stars = Math.round(v);
  buildStars(state.stars);
  persistTweaks({ stars: state.stars });
});
bindSlider('size-slider', 'size-val', (v) => {
  state.nodeSize = v;
  nodeMaterial.uniforms.u_sizeScale.value = v;
  persistTweaks({ nodeSize: v });
});
bindSlider('pulse-slider', 'pulse-val', (v) => {
  state.pulseSpeed = v;
  edgeMaterial.uniforms.u_pulseSpeed.value = v;
  persistTweaks({ pulseSpeed: v });
});
bindSlider('bloom-slider', 'bloom-val', (v) => {
  state.bloom = v;
  bloomPass.strength = v;
  persistTweaks({ bloom: v });
});
bindSlider('bloom-radius-slider', 'bloom-radius-val', (v) => {
  state.bloomRadius = v;
  bloomPass.radius = v;
  persistTweaks({ bloomRadius: v });
});
bindSlider('edge-slider', 'edge-val', (v) => {
  state.edgeOpacity = v;
  applyEdgeFilters();
  persistTweaks({ edgeOpacity: v });
});

// Pre-compute edge lengths for range filter
const edgeLengths = new Float32Array(edgeCount);
let maxEdgeLength = 0;
graph.edges.forEach(([a, b], i) => {
  const na = graph.nodes[a], nb = graph.nodes[b];
  const dx = na.pos.x - nb.pos.x, dy = na.pos.y - nb.pos.y, dz = na.pos.z - nb.pos.z;
  edgeLengths[i] = Math.sqrt(dx*dx + dy*dy + dz*dz);
  if (edgeLengths[i] > maxEdgeLength) maxEdgeLength = edgeLengths[i];
});

function applyEdgeFilters() {
  refreshEdgeAlphas();
}

bindSlider('edgerange-slider', 'edgerange-val', (v) => {
  state.edgeRange = v;
  document.getElementById('edgerange-val').textContent = Math.round(v * 100) + '%';
  applyEdgeFilters();
  persistTweaks({ edgeRange: v });
}, true);
bindSlider('edgewidth-slider', 'edgewidth-val', (v) => {
  edgeLines.material.linewidth = v;
  persistTweaks({ edgeWidth: v });
});
bindSlider('intra-slider', 'intra-val', (v) => {
  state.intraOnly = v >= 1;
  document.getElementById('intra-val').textContent = state.intraOnly ? 'AN' : 'AUS';
  applyEdgeFilters();
  persistTweaks({ intraOnly: state.intraOnly });
}, true);

// init slider UI values
document.getElementById('glow-slider').value = state.glow;
document.getElementById('stars-slider').value = state.stars;
document.getElementById('size-slider').value = state.nodeSize;
document.getElementById('pulse-slider').value = state.pulseSpeed;
document.getElementById('bloom-slider').value = state.bloom;
document.getElementById('bloom-radius-slider').value = state.bloomRadius;
document.getElementById('edge-slider').value = state.edgeOpacity;
document.getElementById('edgerange-slider').value = state.edgeRange;
document.getElementById('edgewidth-slider').value = TWEAK_DEFAULTS.edgeWidth || 1.0;
edgeLines.material.linewidth = TWEAK_DEFAULTS.edgeWidth || 1.0;
document.getElementById('intra-slider').value = state.intraOnly ? 1 : 0;
// Apply edge filters on startup so alphas match slider values
applyEdgeFilters();
document.getElementById('glow-val').textContent = state.glow.toFixed(2);
document.getElementById('stars-val').textContent = state.stars;
document.getElementById('size-val').textContent = state.nodeSize.toFixed(1);
document.getElementById('pulse-val').textContent = state.pulseSpeed.toFixed(2);
document.getElementById('bloom-val').textContent = state.bloom.toFixed(2);
document.getElementById('bloom-radius-val').textContent = state.bloomRadius.toFixed(2);
document.getElementById('edge-val').textContent = state.edgeOpacity.toFixed(2);
document.getElementById('edgerange-val').textContent = Math.round(state.edgeRange * 100) + '%';
document.getElementById('intra-val').textContent = state.intraOnly ? 'AN' : 'AUS';

// Top buttons
document.getElementById('btn-rotate').addEventListener('click', () => {
  controls.autoRotate = !controls.autoRotate;
  document.getElementById('btn-rotate').style.color = controls.autoRotate ? 'var(--accent)' : '';
});
document.getElementById('btn-rotate').style.color = 'var(--accent)';

// Heatmap toggle
const btnHeat = document.getElementById('btn-heat');
btnHeat.addEventListener('click', () => {
  state.heatmap = !state.heatmap;
  btnHeat.classList.toggle('heat-active', state.heatmap);
  if (state.heatmap) {
    // Seed some baseline usage so it's not empty on first toggle
    if (regionUsage.every(v => v === 0)) {
      // deterministic-ish baseline: hub-heavy regions "feel" hotter
      regionUsage[0] = 8;   // prefrontal
      regionUsage[3] = 12;  // hippo
      regionUsage[8] = 14;  // thalamus
      regionUsage[5] = 6;   // nucleus
      regionUsage[1] = 10;  // motor
      regionUsage[10] = 5;  // basal
      regionUsage[7] = 4;   // visual
      regionUsage[6] = 3;   // broca
      regionUsage[2] = 3;   // sensory
      regionUsage[4] = 2;   // cerebellum
      regionUsage[11] = 2;  // amygdala
      regionUsage[9] = 1;   // stem
    }
    applyHeatmap();
    termLine('SYS', 'Heatmap: <span class="hl">ON</span> · zeigt Claude-Nutzung pro Region', { tagClass: 'tag-mem', out: true });
  } else {
    clearHeatmap();
    termLine('SYS', 'Heatmap: <span class="hl">OFF</span>', { tagClass: 'tag-mem', out: true });
  }
});

document.getElementById('btn-reset').addEventListener('click', () => {
  controls.target.set(0, 0, 0);
  camera.position.set(0, 14, 110);
});
document.getElementById('btn-obsidian').addEventListener('click', () => {
  if (state.selectedId != null) {
    const n = graph.nodes[state.selectedId];
    if (window.__onNodeClick) {
      window.__onNodeClick(state.selectedId, n.title);
    }
  }
});

// Tweaks panel toggle (also wired to host editmode)
const tweaksPanel = document.getElementById('tweaks-panel');
document.getElementById('btn-tweaks').addEventListener('click', () => tweaksPanel.classList.toggle('visible'));
document.getElementById('tweaks-close').addEventListener('click', () => tweaksPanel.classList.remove('visible'));

// ============ SETTINGS PANEL ============
const settingsPanel = document.getElementById('settings-panel');
const settingsBackdrop = document.getElementById('settings-backdrop');

function openSettings() {
  settingsPanel.classList.add('visible');
  settingsBackdrop.classList.add('visible');
  loadSettings();
}
function closeSettings() {
  settingsPanel.classList.remove('visible');
  settingsBackdrop.classList.remove('visible');
}

document.getElementById('btn-settings').addEventListener('click', openSettings);
document.getElementById('settings-close').addEventListener('click', closeSettings);
settingsBackdrop.addEventListener('click', closeSettings);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && settingsPanel.classList.contains('visible')) closeSettings();
});

const SETTINGS_REGION_NAMES = [
  'Praefrontaler Cortex','Motorischer Cortex','Sensorischer Cortex',
  'Hippocampus','Kleinhirn','Nucleus Accumbens','Broca-Areal',
  'Visueller Cortex','Thalamus','Stammhirn','Basalganglien','Amygdala',
];

function buildMappingRow(folder, idx) {
  const regionName = SETTINGS_REGION_NAMES[idx] || `Region ${idx}`;
  const color = PALETTES[state.palette][idx] || '#5EE9F0';
  const row = document.createElement('div');
  row.className = 'mapping-row';

  const dot = document.createElement('span');
  dot.className = 'mapping-dot';
  dot.style.background = color;
  dot.style.boxShadow = `0 0 6px ${color}`;

  const folderSpan = document.createElement('span');
  folderSpan.className = 'mapping-folder';
  folderSpan.textContent = folder;

  const arrow = document.createElement('span');
  arrow.className = 'mapping-arrow';
  arrow.textContent = '\u2192';

  const regionSpan = document.createElement('span');
  regionSpan.className = 'mapping-region';
  regionSpan.textContent = regionName;

  row.append(dot, folderSpan, arrow, regionSpan);
  return row;
}

async function loadSettings() {
  try {
    const [cfgRes, statsRes] = await Promise.all([
      fetch('/api/config'),
      fetch('/api/stats'),
    ]);
    if (!cfgRes.ok || !statsRes.ok) throw new Error('API error: ' + cfgRes.status);
    const cfg = await cfgRes.json();
    const stats = await statsRes.json();

    // Vault path
    const vaultEl = document.getElementById('cfg-vault-path');
    if (cfg.vault_path) {
      const parts = cfg.vault_path.replace(/[\\/]+/g, '/').split('/');
      vaultEl.textContent = parts.length > 3
        ? '.../' + parts.slice(-3).join('/')
        : cfg.vault_path;
      vaultEl.title = cfg.vault_path;
    } else {
      vaultEl.textContent = '(nicht konfiguriert)';
      vaultEl.title = '';
    }

    // Model
    document.getElementById('cfg-model').textContent = cfg.model_name || '\u2014';

    // Toggles
    document.getElementById('cfg-auto-index').checked = cfg.auto_index;
    document.getElementById('cfg-index-startup').checked = cfg.index_on_startup;

    // Log level
    document.getElementById('cfg-log-level').value = cfg.log_level || 'INFO';

    // Folder-to-region mapping
    const mappingsEl = document.getElementById('cfg-mappings');
    mappingsEl.replaceChildren();
    const entries = Object.entries(cfg.folder_to_region || {});
    if (entries.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'mapping-empty';
      empty.textContent = 'Kein Mapping konfiguriert';
      mappingsEl.appendChild(empty);
    } else {
      entries.sort((a, b) => a[0].localeCompare(b[0]));
      for (const [folder, idx] of entries) {
        mappingsEl.appendChild(buildMappingRow(folder, idx));
      }
    }

    // Stats
    document.getElementById('cfg-notes').textContent = stats.notes?.toLocaleString() || '0';
    document.getElementById('cfg-vectors').textContent = stats.vectors?.toLocaleString() || '0';
    document.getElementById('cfg-regions').textContent = stats.regions || '12';
    document.getElementById('cfg-edges').textContent = stats.edges?.toLocaleString() || '0';
  } catch (e) {
    console.error('Settings load failed:', e);
  }
}

async function saveSettingField(key, value, revertFn) {
  try {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
  } catch (e) {
    console.error('Settings save failed:', e);
    if (revertFn) revertFn();
  }
}

// Bind toggle switches
document.getElementById('cfg-auto-index').addEventListener('change', (e) => {
  const prev = !e.target.checked;
  saveSettingField('auto_index', e.target.checked, () => { e.target.checked = prev; });
});
document.getElementById('cfg-index-startup').addEventListener('change', (e) => {
  const prev = !e.target.checked;
  saveSettingField('index_on_startup', e.target.checked, () => { e.target.checked = prev; });
});
document.getElementById('cfg-log-level').addEventListener('change', (e) => {
  const prev = e.target.value;
  saveSettingField('log_level', e.target.value, () => { e.target.value = prev; });
});

// Reindex button
const reindexBtn = document.getElementById('cfg-reindex');
const reindexBtnOriginal = reindexBtn.cloneNode(true);
reindexBtn.addEventListener('click', async () => {
  const statusEl = document.getElementById('cfg-reindex-status');
  reindexBtn.classList.add('loading');
  reindexBtn.textContent = 'Indexing...';
  statusEl.textContent = '';
  try {
    const res = await fetch('/api/reindex', { method: 'POST' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (data.error) {
      statusEl.textContent = 'Fehler: ' + data.error;
      statusEl.style.color = 'var(--danger)';
    } else {
      statusEl.textContent = data.indexed + ' Notes indexiert in ' + data.elapsed + 's';
      statusEl.style.color = 'var(--accent)';
      loadSettings();
    }
  } catch (e) {
    statusEl.textContent = 'Fehler: ' + e.message;
    statusEl.style.color = 'var(--danger)';
  } finally {
    reindexBtn.classList.remove('loading');
    reindexBtn.replaceChildren(...reindexBtnOriginal.cloneNode(true).childNodes);
  }
});

// Detail panel
const detailPanel = document.getElementById('detail-panel');
const detailEmpty = document.getElementById('detail-empty');
function renderDetail(node) {
  const r = REGIONS[node.regionIdx];
  const color = PALETTES[state.palette][node.regionIdx];
  const neighbors = nodeEdges[node.id]
    .map(ei => {
      const e = graph.edges[ei];
      return e[0] === node.id ? e[1] : e[0];
    })
    .slice(0, 8)
    .map(nid => graph.nodes[nid]);

  detailPanel.innerHTML = `
    <div class="detail-header">
      <div class="detail-kicker">
        <span class="k-dot" style="background:${color};color:${color}"></span>
        ${r.name}
      </div>
      <div class="detail-title">${esc(node.title)}</div>
      <div class="detail-meta">
        <span><strong>${node.degree}</strong>verbindungen</span>
        <span><strong>${node.wordCount}</strong>wörter</span>
        <span><strong>${node.created.toLocaleDateString('de-DE',{month:'short',day:'2-digit'})}</strong></span>
      </div>
    </div>
    <div class="detail-section">
      <h3>Tags</h3>
      <div class="tag-list">${node.tags.map(t => `<span class="tag">${esc(t)}</span>`).join('')}</div>
    </div>
    <div class="detail-section">
      <h3>Verknüpfte Notizen (${neighbors.length})</h3>
      <div class="link-list">
        ${neighbors.map(nn => {
          const nc = PALETTES[state.palette][nn.regionIdx];
          return `<div class="link-item" data-nid="${nn.id}">
            <span class="li-dot" style="background:${nc};color:${nc}"></span>
            <span class="li-title">${esc(nn.title)}</span>
            <span class="li-arrow">→</span>
          </div>`;
        }).join('')}
      </div>
    </div>
    <div class="detail-actions">
      <button class="btn primary" id="detail-open">
        In Obsidian öffnen
      </button>
      <button class="btn" id="detail-close">Schließen</button>
    </div>
  `;
  detailPanel.querySelectorAll('.link-item').forEach(el => {
    el.addEventListener('click', () => selectNode(parseInt(el.dataset.nid)));
  });
  detailPanel.querySelector('#detail-open').addEventListener('click', () => {
    if (window.__onNodeClick) window.__onNodeClick(node.id, node.title);
  });
  detailPanel.querySelector('#detail-close').addEventListener('click', clearSelection);
}
function renderDetailEmpty() {
  detailPanel.innerHTML = '';
  detailPanel.appendChild(detailEmpty);
}

// ===== SEARCH — local node search with visual filtering =====
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
const searchWrap = document.getElementById('search-wrap');
state.searchFilter = null; // Set<nodeId> when filter is active

function searchNodes(query) {
  const q = query.toLowerCase();
  const tokens = q.split(/\s+/).filter(x => x.length > 1);
  if (!tokens.length) return [];
  const scored = [];
  for (let i = 0; i < graph.nodes.length; i++) {
    const n = graph.nodes[i];
    const title = n.title.toLowerCase();
    const tagStr = (n.tags || []).join(' ').toLowerCase();
    const region = REGIONS[n.regionIdx].name.toLowerCase();
    const path = (n.source_file || '').toLowerCase();
    let score = 0;
    for (const t of tokens) {
      if (title === t) score += 5;
      else if (title.startsWith(t)) score += 3;
      else if (title.includes(t)) score += 2;
      if (path.includes(t)) score += 1.5;
      if (tagStr.includes(t)) score += 1;
      if (region.includes(t)) score += 0.5;
    }
    if (score > 0) scored.push({ node: n, score, id: i });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 30);
}

function renderSearchResults(results, query) {
  // Clear previous
  while (searchResults.firstChild) searchResults.removeChild(searchResults.firstChild);
  if (!results.length || !query) {
    searchResults.classList.remove('visible');
    return;
  }
  const palette = PALETTES[state.palette];
  const isFiltered = state.searchFilter !== null;

  // Header
  const head = document.createElement('div');
  head.className = 'sr-head';
  const headLabel = document.createElement('span');
  headLabel.textContent = results.length + ' Ergebnisse';
  head.appendChild(headLabel);
  const filterBtn = document.createElement('button');
  filterBtn.className = 'sr-filter-btn' + (isFiltered ? ' active' : '');
  filterBtn.textContent = isFiltered ? '\u2715 Filter aus' : '\u2295 Nur diese';
  filterBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (state.searchFilter) {
      clearSearchFilter();
    } else {
      applySearchFilter(new Set(results.map(r => r.id)));
    }
    renderSearchResults(results, query);
  });
  head.appendChild(filterBtn);
  searchResults.appendChild(head);

  // Result items
  for (const r of results) {
    const col = palette[r.node.regionIdx];
    const rName = REGIONS[r.node.regionIdx].name;
    const item = document.createElement('div');
    item.className = 'sr-item';
    item.dataset.nodeId = r.id;
    const dot = document.createElement('span');
    dot.className = 'sr-dot';
    dot.style.background = col;
    dot.style.color = col;
    const titleEl = document.createElement('span');
    titleEl.className = 'sr-title';
    titleEl.textContent = r.node.title;
    const regionEl = document.createElement('span');
    regionEl.className = 'sr-region';
    regionEl.textContent = rName;
    item.appendChild(dot);
    item.appendChild(titleEl);
    item.appendChild(regionEl);
    item.addEventListener('click', () => {
      flyToNode(r.id);
      selectNode(r.id);
    });
    searchResults.appendChild(item);
  }
  searchResults.classList.add('visible');
}

function applySearchFilter(hitIds) {
  state.searchFilter = hitIds;
  searchWrap.classList.add('has-filter');
  for (let i = 0; i < graph.nodes.length; i++) {
    nodeAlpha[i] = hitIds.has(i) ? 1.0 : 0.04;
  }
  nodeGeometry.attributes.alpha.needsUpdate = true;
  graph.edges.forEach(([a, b], i) => {
    const show = hitIds.has(a) && hitIds.has(b);
    const same = graph.nodes[a].region === graph.nodes[b].region;
    const baseA = show ? (same ? 0.5 : 0.25) : 0.0;
    const off = i * vertsPerEdge;
    for (let s = 0; s < segPerEdge; s++) {
      const t0 = s / segPerEdge;
      const t1 = (s + 1) / segPerEdge;
      const tap = (x) => Math.pow(Math.sin(x * Math.PI), 0.6);
      const vi = off + s * 2;
      edgeAlphas[vi]   = baseA * tap(t0);
      edgeAlphas[vi+1] = baseA * tap(t1);
    }
  });
  edgeGeom.attributes.alpha.needsUpdate = true;
  updateStats();
}

function clearSearchFilter() {
  state.searchFilter = null;
  searchWrap.classList.remove('has-filter');
  for (let i = 0; i < graph.nodes.length; i++) {
    nodeAlpha[i] = (!state.activeRegion || graph.nodes[i].region === state.activeRegion) ? 1.0 : 0.06;
  }
  nodeGeometry.attributes.alpha.needsUpdate = true;
  applyEdgeFilters();
  updateStats();
}

let flyT = null;
function flyToNode(id) {
  const n = graph.nodes[id];
  const startTarget = controls.target.clone();
  const startCam = camera.position.clone();
  const endTarget = n.pos.clone();
  const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
  const endCam = n.pos.clone().add(dir.multiplyScalar(45));
  const t0 = performance.now();
  if (flyT) cancelAnimationFrame(flyT);
  function step() {
    const k = Math.min(1, (performance.now() - t0) / 700);
    const ease = 1 - Math.pow(1 - k, 3);
    controls.target.lerpVectors(startTarget, endTarget, ease);
    camera.position.lerpVectors(startCam, endCam, ease);
    if (k < 1) flyT = requestAnimationFrame(step);
  }
  step();
}

let _searchDebounce = null;
searchInput.addEventListener('input', () => {
  clearTimeout(_searchDebounce);
  _searchDebounce = setTimeout(() => {
    const q = searchInput.value.trim();
    if (!q) {
      searchResults.classList.remove('visible');
      if (state.searchFilter) clearSearchFilter();
      else {
        for (let i = 0; i < graph.nodes.length; i++) nodeAlpha[i] = 1.0;
        nodeGeometry.attributes.alpha.needsUpdate = true;
      }
      return;
    }
    const results = searchNodes(q);
    renderSearchResults(results, q);
    if (!state.searchFilter) {
      const hitIds = new Set(results.map(r => r.id));
      for (let i = 0; i < graph.nodes.length; i++) {
        nodeAlpha[i] = hitIds.has(i) ? 1.0 : 0.15;
      }
      nodeGeometry.attributes.alpha.needsUpdate = true;
    }
  }, 120);
});

searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && searchInput.value.trim()) {
    const results = searchNodes(searchInput.value.trim());
    if (results.length) {
      applySearchFilter(new Set(results.map(r => r.id)));
      renderSearchResults(results, searchInput.value.trim());
    }
  }
});

document.addEventListener('click', (e) => {
  if (!searchWrap.contains(e.target)) {
    searchResults.classList.remove('visible');
    if (!state.searchFilter && searchInput.value.trim()) {
      for (let i = 0; i < graph.nodes.length; i++) nodeAlpha[i] = 1.0;
      nodeGeometry.attributes.alpha.needsUpdate = true;
    }
  }
});

searchInput.addEventListener('focus', () => {
  if (searchInput.value.trim()) {
    const results = searchNodes(searchInput.value.trim());
    renderSearchResults(results, searchInput.value.trim());
  }
});

window.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    searchInput.focus();
    searchInput.select();
  }
  if (e.key === 'Escape') {
    if (state.searchFilter) {
      clearSearchFilter();
      searchInput.value = '';
      searchResults.classList.remove('visible');
    } else if (searchInput.value) {
      searchInput.value = '';
      searchResults.classList.remove('visible');
      for (let i = 0; i < graph.nodes.length; i++) nodeAlpha[i] = 1.0;
      nodeGeometry.attributes.alpha.needsUpdate = true;
    } else {
      clearSelection();
    }
  }
});

// Stats
function updateStats() {
  const n = state.activeRegion
    ? graph.nodes.filter(x => x.region === state.activeRegion).length
    : graph.nodes.length;
  const e = state.activeRegion
    ? graph.edges.filter(([a,b]) => graph.nodes[a].region === state.activeRegion && graph.nodes[b].region === state.activeRegion).length
    : graph.edges.length;
  document.getElementById('active-nodes').textContent = n;
  document.getElementById('active-edges').textContent = e;
  document.getElementById('avg-degree').textContent = (e * 2 / Math.max(n,1)).toFixed(1);
  document.getElementById('hud-nodes').textContent = n;
  document.getElementById('hud-edges').textContent = e;
  document.getElementById('st-graph').textContent = `${n} NODES · ${e} EDGES`;
}
updateStats();

// Recent Claude queries
const recentEl = document.getElementById('recent-queries');
const SAMPLE_QUERIES = [
  { q: 'Wie ist die Architektur von Neural-Brain?', regions: ['prefrontal','hippo','visual'], ts: '2m' },
  { q: 'Letzter Commit am GL-Widget?', regions: ['motor','sensory'], ts: '14m' },
  { q: 'Welche MCP-Endpoints nutze ich?', regions: ['thalamus','stem'], ts: '1h' },
  { q: 'Offene Bugs & Risiken?', regions: ['amygdala','motor'], ts: '3h' },
  { q: 'Wochenziele diese Woche', regions: ['nucleus','prefrontal'], ts: 'gestern' },
  { q: 'Schreibstil für Prompt-Engineering-Doc', regions: ['broca','visual'], ts: '2d' },
  { q: 'Welche Tools nutze ich für Deploy?', regions: ['basal','motor','cerebellum'], ts: '3d' },
  { q: 'Pattern aus letzter Design-Review', regions: ['visual','hippo'], ts: '1w' },
];
SAMPLE_QUERIES.forEach(s => {
  const row = document.createElement('div');
  row.className = 'qitem';
  row.innerHTML = `
    <div class="q-text">${s.q}</div>
    <div class="q-meta"><span>${s.ts}</span><span>${s.regions.length} REGIONEN</span><span class="q-score">${(0.7 + Math.random()*0.28).toFixed(2)}</span></div>
  `;
  row.addEventListener('click', () => { searchInput.value = s.q; runClaudeQuery(s.q); });
  recentEl.appendChild(row);
});

/* ========================================================================
   REGION LABEL OVERLAYS (projected 3D → 2D)
   ======================================================================== */

const regionLabelsEl = document.getElementById('region-labels');
const regionLabels = REGIONS.map((r, i) => {
  const el = document.createElement('div');
  el.className = 'region-label';
  const c = PALETTES[state.palette][i];
  el.innerHTML = `<span class="rl-line"></span><span class="rl-dot" style="background:${c};color:${c}"></span>${r.name}`;
  regionLabelsEl.appendChild(el);
  return el;
});

let _cachedRect = null;
let _labelFrame = 0;
function _updateCachedRect() { _cachedRect = canvas.getBoundingClientRect(); }
window.addEventListener('resize', _updateCachedRect);
_updateCachedRect();

function updateRegionLabels() {
  _labelFrame++;
  if (_labelFrame % 3 !== 0) return;
  if (!_cachedRect) _updateCachedRect();
  const rect = _cachedRect;
  const vec = new THREE.Vector3();
  REGIONS.forEach((r, i) => {
    vec.set(r.pos[0], r.pos[1], r.pos[2]).project(camera);
    if (vec.z > 1) { regionLabels[i].classList.remove('visible'); return; }
    const x = (vec.x * 0.5 + 0.5) * rect.width;
    const y = (-vec.y * 0.5 + 0.5) * rect.height;
    regionLabels[i].style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
    const dim = state.activeRegion && state.activeRegion !== r.key;
    regionLabels[i].style.color = dim ? 'rgba(94,106,131,0.4)' : (PALETTES[state.palette][i] || '#E6EEFB');
    regionLabels[i].classList.add('visible');
  });
}

/* ========================================================================
   ANIMATION LOOP
   ======================================================================== */

const clock = new THREE.Clock();
let fpsTime = 0, fpsFrames = 0;
const fpsEl = document.getElementById('hud-fps');
const camEl = document.getElementById('hud-cam');

/* ========================================================================
   INTELLIGENCE LAYER — Claude query beams, cross-region arcs, minimap,
   timeline, brain boot, MCP HUD
   ======================================================================== */

// ----- Claude query system: retrieval beam + injected-context chip -----
const chipQuery = document.getElementById('chip-query');
const chipQueryText = document.getElementById('chip-query-text');
const ctxCountEl = document.getElementById('ctx-count');
const mcpReqEl = document.getElementById('mcp-req');
const injectedEl = document.getElementById('injected');
const injectedPillsEl = document.getElementById('injected-pills');
const ctBody = document.getElementById('ct-body');
const ctStatus = document.getElementById('ct-status');
const ctHead = document.getElementById('ct-head');
const claudeTerm = document.getElementById('claude-term');

let mcpRequestCount = 0;

// ----- Hot/cold heatmap overlay -----
// Tints the scene with a red/blue overlay based on per-region usage frequency.
// Uses nodeAlpha + an auxiliary color-ramp applied to edges via existing palette system.
function applyHeatmap() {
  if (!state.heatmap) return;
  // Normalize region usage
  const maxUsage = Math.max(1, ...regionUsage);
  const c = new THREE.Color();
  graph.nodes.forEach((n, i) => {
    const u = regionUsage[n.regionIdx] / maxUsage;
    // cold -> hot ramp (slate → red)
    // u=0: #2A3446 (cold slate); u=1: #FF5C7C (hot)
    const r = 0.165 + (1.0   - 0.165) * u;
    const g = 0.204 + (0.361 - 0.204) * u;
    const b = 0.275 + (0.486 - 0.275) * u;
    nodeColors[i*3]     = r;
    nodeColors[i*3 + 1] = g;
    nodeColors[i*3 + 2] = b;
  });
  nodeGeometry.attributes.color.needsUpdate = true;
}

function clearHeatmap() {
  applyPaletteToNodes();
}

// Collapse terminal on head click
ctHead.addEventListener('click', (e) => {
  if (e.target.closest('#ct-resize')) return;
  claudeTerm.classList.toggle('expanded');
});

// Resize terminal by dragging bottom-right handle
const ctResize = document.getElementById('ct-resize');
let rs = null;
ctResize.addEventListener('mousedown', (e) => {
  e.preventDefault();
  e.stopPropagation();
  const rect = claudeTerm.getBoundingClientRect();
  rs = { x0: e.clientX, y0: e.clientY, w0: rect.width, h0: rect.height };
  claudeTerm.classList.add('resizing');
  claudeTerm.classList.add('expanded');
  claudeTerm.style.maxHeight = rect.height + 'px';
});
window.addEventListener('mousemove', (e) => {
  if (!rs) return;
  const w = Math.max(220, Math.min(window.innerWidth - 40, rs.w0 + (e.clientX - rs.x0)));
  const h = Math.max(60,  Math.min(window.innerHeight - 80, rs.h0 + (e.clientY - rs.y0)));
  claudeTerm.style.width = w + 'px';
  claudeTerm.style.height = h + 'px';
  claudeTerm.style.maxHeight = h + 'px';
});
window.addEventListener('mouseup', () => {
  if (rs) { rs = null; claudeTerm.classList.remove('resizing'); }
});

// Chat input inside terminal
const ctInput = document.getElementById('ct-input');
const ctHistory = [];
let ctHistIdx = -1;

function focusTerminalInput() {
  if (claudeTerm.classList.contains('expanded')) {
    setTimeout(() => ctInput.focus(), 50);
  }
}

ctInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const q = ctInput.value.trim();
    if (!q) return;
    ctHistory.push(q);
    ctHistIdx = ctHistory.length;
    ctInput.value = '';
    runClaudeQuery(q);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    if (ctHistIdx > 0) { ctHistIdx--; ctInput.value = ctHistory[ctHistIdx] || ''; }
  } else if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (ctHistIdx < ctHistory.length - 1) { ctHistIdx++; ctInput.value = ctHistory[ctHistIdx] || ''; }
    else { ctHistIdx = ctHistory.length; ctInput.value = ''; }
  }
});

// Auto-focus input when terminal is expanded (not when resizing)
ctHead.addEventListener('click', (e) => {
  if (e.target.closest('#ct-resize')) return;
  // toggle already happened above; focus if just expanded
  setTimeout(() => {
    if (claudeTerm.classList.contains('expanded')) ctInput.focus();
  }, 100);
});

function termLine(tag, text, opts = {}) {
  const div = document.createElement('div');
  div.className = 'ct-line';
  const tagEl = document.createElement('span');
  tagEl.className = 'ct-tag ' + (opts.tagClass || 'tag-tool');
  tagEl.textContent = tag;
  const textEl = document.createElement('span');
  textEl.className = 'ct-text' + (opts.out ? ' out' : '');
  textEl.innerHTML = text;
  div.appendChild(tagEl);
  div.appendChild(textEl);
  ctBody.appendChild(div);
  ctBody.scrollTop = ctBody.scrollHeight;
  // Keep max ~60 lines
  while (ctBody.children.length > 60) ctBody.removeChild(ctBody.firstChild);
  return textEl;
}

function setCtStatus(state) {
  if (state === 'thinking') {
    ctStatus.textContent = '● THINKING';
    ctStatus.classList.add('thinking');
  } else if (state === 'retrieving') {
    ctStatus.textContent = '● RETRIEVING';
    ctStatus.classList.add('thinking');
  } else {
    ctStatus.textContent = '● IDLE';
    ctStatus.classList.remove('thinking');
  }
}

// Per-node "recency" (time since last Claude-read); decays on each frame.
const nodeRecency = new Float32Array(graph.nodes.length);
// Per-node lifetime usage — for the hot/cold heatmap
const nodeUsage = new Float32Array(graph.nodes.length);
const regionUsage = new Float32Array(REGIONS.length);

// Active cross-region arcs: list of {aIdx, bIdx, t0, dur, color}
const activeArcs = [];
const arcGroup = new THREE.Group();
scene.add(arcGroup);

function sampleRelevantNodes(query) {
  // Simple fake "retrieval": rank nodes by token match + tag overlap, take top-12.
  const q = query.toLowerCase();
  const tokens = q.split(/\s+/).filter(x => x.length > 2);
  const scored = graph.nodes.map(n => {
    const hay = (n.title + ' ' + (n.tags || []).join(' ')).toLowerCase();
    let s = 0;
    tokens.forEach(t => { if (hay.includes(t)) s += 1; });
    s += Math.random() * 0.4; // jitter so repeated queries look alive
    if (n.hub) s += 0.15;
    return { n, s };
  });
  scored.sort((a,b) => b.s - a.s);
  return scored.slice(0, 12).map(x => x.n);
}

function bumpMcpReq() {
  mcpRequestCount++;
  mcpReqEl.textContent = `·${mcpRequestCount} req`;
}

function renderInjectedPills(hits) {
  injectedPillsEl.innerHTML = '';
  const palette = PALETTES[state.palette];
  hits.forEach((n, i) => {
    setTimeout(() => {
      const pill = document.createElement('div');
      pill.className = 'inj-pill';
      const col = palette[n.regionIdx];
      const title = n.title.length > 22 ? n.title.slice(0, 22) + '…' : n.title;
      pill.innerHTML = `<span class="ip-dot" style="background:${col};color:${col}"></span>${title}`;
      injectedPillsEl.appendChild(pill);
    }, i * 85);
  });
  injectedEl.classList.add('visible');
}

function hideInjectedPills() {
  injectedEl.classList.remove('visible');
}

function runClaudeQuery(query) {
  const hits = sampleRelevantNodes(query);
  if (!hits.length) return;

  bumpMcpReq();
  setCtStatus('retrieving');
  // Auto-expand terminal so the user sees activity
  claudeTerm.classList.add('expanded');

  // Terminal log
  termLine('> USER', query, { tagClass: 'tag-you' });
  termLine('TOOL', `brain_retrieve({ q: "${query.length > 40 ? query.slice(0,40)+'…' : query}" })`, { tagClass: 'tag-tool' });
  termLine('MCP', `POST gystc://retrieve · <span class="hl">${hits.length} nodes</span>`, { tagClass: 'tag-mem' });

  // Chip state
  chipQuery.style.display = '';
  chipQueryText.textContent = `RETRIEVING · „${query.length > 28 ? query.slice(0,28)+'…' : query}"`;
  ctxCountEl.textContent = hits.length;

  // Record usage for heatmap
  hits.forEach(n => {
    nodeUsage[n.id] += 1;
    regionUsage[n.regionIdx] += 1;
  });

  // Beam walk: light up nodes sequentially like a retrieval cascade
  hits.forEach((n, i) => {
    setTimeout(() => {
      nodeRecency[n.id] = 1.0;            // strong glow
      nodeAlpha[n.id] = 1.0;
      nodeGeometry.attributes.alpha.needsUpdate = true;
    }, i * 85);
  });

  renderInjectedPills(hits);

  // Cross-region arcs between hit pairs in different regions (top signal)
  const byRegion = {};
  hits.forEach(n => { (byRegion[n.region] ||= []).push(n); });
  const regionKeys = Object.keys(byRegion);
  for (let i = 0; i < regionKeys.length - 1; i++) {
    const a = byRegion[regionKeys[i]][0];
    const b = byRegion[regionKeys[i + 1]][0];
    if (a && b) spawnArc(a, b, 2.4);
  }

  // Dim everything else
  const hitIds = new Set(hits.map(n => n.id));
  for (let i = 0; i < graph.nodes.length; i++) {
    if (!hitIds.has(i)) nodeAlpha[i] = 0.12;
  }
  nodeGeometry.attributes.alpha.needsUpdate = true;

  // After retrieval, go into "thinking" state and attempt a real Claude completion
  setTimeout(async () => {
    chipQueryText.textContent = `INJECTED · ${hits.length} NODES`;
    for (let i = 0; i < graph.nodes.length; i++) {
      if (!hitIds.has(i)) nodeAlpha[i] = 0.7;
    }
    nodeGeometry.attributes.alpha.needsUpdate = true;

    setCtStatus('thinking');
    const thinkingLine = termLine('CLAUDE', '<span class="ct-caret"></span>', { tagClass: 'tag-out', out: true });

    // Build a grounded prompt from the hits
    const ctx = hits.slice(0, 8).map(n => `- ${n.title} [${REGIONS[n.regionIdx].name}]`).join('\n');
    const prompt = `Du bist Claude Code mit Zugriff auf meinen Obsidian-Vault. Nutze NUR diesen abgerufenen Kontext:\n\n${ctx}\n\nFrage: ${query}\n\nAntworte auf Deutsch, max. 2 kurze Sätze, referenziere die relevantesten Notizen per Namen. Keine Einleitung.`;

    let answer;
    try {
      answer = await Promise.race([
        window.claude?.complete(prompt),
        new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 8000)),
      ]);
    } catch (err) {
      // Fallback if claude unavailable
      const top = hits.slice(0, 3).map(n => `<span class="hl">${esc(n.title)}</span>`).join(', ');
      answer = `Basierend auf ${top} — ${REGIONS[hits[0].regionIdx].subtitle.toLowerCase()}.`;
    }
    thinkingLine.textContent = answer || '(keine Antwort)';
    setCtStatus('idle');
  }, hits.length * 85 + 400);

  setTimeout(() => {
    chipQuery.style.display = 'none';
    hideInjectedPills();
    for (let i = 0; i < graph.nodes.length; i++) nodeAlpha[i] = 1.0;
    nodeGeometry.attributes.alpha.needsUpdate = true;
    // Refresh heatmap if active
    if (state.heatmap) applyHeatmap();
  }, hits.length * 85 + 9200);
}

function spawnArc(a, b, dur = 2.0) {
  const palette = PALETTES[state.palette];
  const col = new THREE.Color(palette[a.regionIdx]).lerp(new THREE.Color(palette[b.regionIdx]), 0.5);

  const midLen = a.pos.distanceTo(b.pos);
  const mid = a.pos.clone().add(b.pos).multiplyScalar(0.5);
  const up = new THREE.Vector3(0, 1, 0);
  const lift = Math.min(midLen * 0.55, 40);
  mid.add(up.multiplyScalar(lift));

  const curve = new THREE.QuadraticBezierCurve3(a.pos.clone(), mid, b.pos.clone());
  const pts = curve.getPoints(48);
  const geom = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({
    color: col, transparent: true, opacity: 0.0,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const line = new THREE.Line(geom, mat);
  arcGroup.add(line);
  activeArcs.push({ line, t0: clock.elapsedTime, dur });
}

function updateArcs(t) {
  for (let i = activeArcs.length - 1; i >= 0; i--) {
    const a = activeArcs[i];
    const k = (t - a.t0) / a.dur; // 0..1
    if (k >= 1) {
      arcGroup.remove(a.line);
      a.line.geometry.dispose();
      a.line.material.dispose();
      activeArcs.splice(i, 1);
      continue;
    }
    // fade in then out, with a shimmer
    const env = Math.sin(k * Math.PI);
    a.line.material.opacity = env * (0.55 + 0.2 * Math.sin(t * 6));
  }
}

// ----- Recency glow uniform (piggyback on existing "activation" attribute) -----
// Lights up nodes Claude is actively reading/writing. Slow decay for visibility.
function updateRecency(dt) {
  let dirty = false;
  for (let i = 0; i < graph.nodes.length; i++) {
    if (nodeRecency[i] > 0) {
      // Fast initial flash (first 0.3s), then slow fade over ~8 seconds
      const rate = nodeRecency[i] > 0.9 ? 0.7 : 0.12;
      nodeRecency[i] = Math.max(0, nodeRecency[i] - dt * rate);
      nodeActivation[i] = nodeRecency[i];
      dirty = true;
    }
  }
  if (dirty) nodeGeometry.attributes.activation.needsUpdate = true;
}

// ----- Minimap -----
const miniCanvas = document.getElementById('minimap-canvas');
const miniCtx = miniCanvas.getContext('2d');
function drawMinimap() {
  const W = miniCanvas.width, H = miniCanvas.height;
  miniCtx.fillStyle = 'rgba(5,7,11,1)';
  miniCtx.fillRect(0, 0, W, H);

  // project node XZ onto mini canvas (top-down)
  const palette = PALETTES[state.palette];
  const scale = 1.6;
  const cx = W / 2, cy = H / 2;
  // faint edges (every 6th)
  miniCtx.globalAlpha = 0.08;
  miniCtx.strokeStyle = '#5EE9F0';
  miniCtx.lineWidth = 0.5;
  miniCtx.beginPath();
  for (let e = 0; e < graph.edges.length; e += 6) {
    const [ai, bi] = graph.edges[e];
    const a = graph.nodes[ai].pos, b = graph.nodes[bi].pos;
    miniCtx.moveTo(cx + a.x * scale, cy + a.z * scale);
    miniCtx.lineTo(cx + b.x * scale, cy + b.z * scale);
  }
  miniCtx.stroke();

  miniCtx.globalAlpha = 1;
  for (let i = 0; i < graph.nodes.length; i++) {
    const n = graph.nodes[i];
    const rec = nodeRecency[i];
    const r = n.hub ? 2.2 : 1.2;
    miniCtx.fillStyle = palette[n.regionIdx];
    miniCtx.globalAlpha = 0.45 + rec * 0.5;
    miniCtx.beginPath();
    miniCtx.arc(cx + n.pos.x * scale, cy + n.pos.z * scale, r + rec * 1.5, 0, Math.PI * 2);
    miniCtx.fill();
  }
  miniCtx.globalAlpha = 1;

  // camera frustum indicator (where you're looking)
  const cam = camera.position;
  miniCtx.strokeStyle = 'rgba(94,233,240,0.9)';
  miniCtx.lineWidth = 1.2;
  miniCtx.beginPath();
  miniCtx.arc(cx + cam.x * scale, cy + cam.z * scale, 4, 0, Math.PI * 2);
  miniCtx.stroke();

  // viewing direction arrow
  const dir = new THREE.Vector3();
  camera.getWorldDirection(dir);
  miniCtx.beginPath();
  miniCtx.moveTo(cx + cam.x * scale, cy + cam.z * scale);
  miniCtx.lineTo(cx + cam.x * scale + dir.x * 18, cy + cam.z * scale + dir.z * 18);
  miniCtx.stroke();
}
let miniTick = 0;

// ----- Timeline scrubber -----
const tlSlider = document.getElementById('timeline-slider');
const tlNow = document.getElementById('tl-now');
const minT = Math.min(...graph.nodes.map(n => n.created.getTime()));
const maxT = Math.max(...graph.nodes.map(n => n.created.getTime()));
tlSlider.addEventListener('input', () => {
  const k = +tlSlider.value / 100;
  const cutoff = minT + (maxT - minT) * k;
  const nodeVisible = new Uint8Array(graph.nodes.length);
  for (let i = 0; i < graph.nodes.length; i++) {
    const v = graph.nodes[i].created.getTime() <= cutoff;
    nodeAlpha[i] = v ? 1.0 : 0.05;
    nodeVisible[i] = v ? 1 : 0;
  }
  nodeGeometry.attributes.alpha.needsUpdate = true;
  for (let ei = 0; ei < graph.edges.length; ei++) {
    const [a, b] = graph.edges[ei];
    const vis = nodeVisible[a] && nodeVisible[b];
    const alpha = vis ? 0.35 : 0.0;
    const off = ei * vertsPerEdge;
    for (let s = 0; s < vertsPerEdge; s++) edgeAlphas[off + s] = alpha;
  }
  edgeGeom.attributes.alpha.needsUpdate = true;
  if (k >= 0.999) {
    tlNow.textContent = 'JETZT';
    tlNow.style.color = 'var(--accent)';
    applyPaletteToEdges();
  } else {
    const d = new Date(cutoff);
    tlNow.textContent = d.toLocaleDateString('de-DE', { month: 'short', year: 'numeric' }).toUpperCase();
    tlNow.style.color = 'var(--ink-1)';
  }
});

// ----- Brain boot: fade overlay, run a demo query once settled -----
const bootEl = document.getElementById('boot');
const bootStatus = document.getElementById('boot-status');
const BOOT_STEPS = [
  'Lade Vault-Index…',
  'Berechne Embeddings…',
  'Propagiere Synapsen…',
  'Verbinde mit MCP…',
  'Gehirn online.'
];
let bootIdx = 0;
const bootTimer = setInterval(() => {
  bootIdx++;
  if (bootIdx < BOOT_STEPS.length) bootStatus.textContent = BOOT_STEPS[bootIdx];
  else {
    clearInterval(bootTimer);
    bootEl.classList.add('hidden');
    setTimeout(() => bootEl.classList.add('removed'), 1300);

    if (window.innerWidth > 1100) claudeTerm.classList.add('expanded');
    termLine('SYS', `GYSTC · <span class="hl">${graph.nodes.length}</span> nodes · <span class="hl">${graph.edges.length}</span> edges`, { tagClass: 'tag-mem', out: true });
    termLine('SYS', 'Warte auf Claude Code Aktivität...', { tagClass: 'tag-mem', out: true });

    // Global bridge: Python activity server pushes events here
    window.addActivityLine = function(tag, text, tagClass) {
      termLine(tag, text, { tagClass: tagClass || 'tag-tool', out: tag === 'CLAUDE' });

      // Fuzzy match: strip HTML tags, extract highlighted term, match by title/stem/substring
      const plainText = text.replace(/<[^>]+>/g, '').toLowerCase();
      const hlMatch = text.match(/<span class='hl'>([^<]+)<\/span>/);
      const hlTerm = hlMatch ? hlMatch[1].toLowerCase().replace(/[_\-]/g, ' ') : '';

      let match = null;
      let bestScore = 0;
      for (const n of graph.nodes) {
        const title = n.title.toLowerCase();
        const stem = title.replace(/[_\-]/g, ' ');
        let score = 0;
        // Exact title match
        if (plainText.includes(title)) score = 3;
        // HL term matches title or stem
        else if (hlTerm && (title.includes(hlTerm) || stem.includes(hlTerm))) score = 2;
        // HL term matches start of any word in title
        else if (hlTerm && stem.split(' ').some(w => w.startsWith(hlTerm))) score = 1;
        if (score > bestScore) { bestScore = score; match = n; }
      }

      if (match) {
        // Fire the matched node
        nodeRecency[match.id] = 1.0;
        nodeUsage[match.id] = Math.min(nodeUsage[match.id] + 0.15, 1.0);
        regionUsage[match.regionIdx] = Math.min(regionUsage[match.regionIdx] + 0.1, 1.0);

        // Synaptic propagation: activate edges (decay naturally via updateEdgeActivation)
        activateEdges(nodeEdges[match.id]);

        // Propagate to direct neighbors (weaker)
        for (const ei of nodeEdges[match.id]) {
          const e = graph.edges[ei];
          const neighborId = e[0] === match.id ? e[1] : e[0];
          nodeRecency[neighborId] = Math.max(nodeRecency[neighborId], 0.35);
        }

        // Spawn pulse ring at the node position
        spawnPulseRing(match);
      }
    };
    window.setTermStatus = function(state) { setCtStatus(state); };
  }
}, 520);
bootStatus.textContent = BOOT_STEPS[0];

// ----- Double-click focus: fly camera to node -----
canvas.addEventListener('dblclick', (e) => {
  const id = pickNode(e.clientX, e.clientY);
  if (id != null) { flyToNode(id); selectNode(id); }
});

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  const t = clock.elapsedTime;
  updateFlyControls();
  controls.update();

  nodeMaterial.uniforms.u_time.value = t;
  edgeMaterial.uniforms.u_time.value = t;
  if (starsObj) starsObj.material.uniforms.u_time.value = t;

  updateRecency(dt);
  updateEdgeActivation(dt);
  updatePulseRings(t);
  updateArcs(t);
  miniTick += dt;
  if (miniTick > 0.12) { drawMinimap(); miniTick = 0; }

  // Rings pulse
  ringGroup.children.forEach((r, i) => {
    r.lookAt(camera.position);
    r.material.opacity = 0.25 + Math.sin(t * 1.2 + i * 0.7) * 0.1;
    r.scale.setScalar(1 + Math.sin(t * 0.9 + i * 0.4) * 0.06);
  });

  updateRegionLabels();

  // Cam + FPS HUD — throttled to reduce DOM thrash
  fpsFrames++;
  fpsTime += dt;
  if (fpsTime > 0.5) {
    fpsEl.textContent = Math.round(fpsFrames / fpsTime);
    const p = camera.position;
    camEl.textContent = `${p.x.toFixed(0)}, ${p.y.toFixed(0)}, ${p.z.toFixed(0)}`;
    fpsFrames = 0; fpsTime = 0;
  }

  composer.render(dt);
}
animate();

/* ========================================================================
   PYTHON DATA BRIDGE
   ======================================================================== */

window.loadGraphData = function(jsonString) {
  window.__graphData = JSON.parse(jsonString);
  // Would need to rebuild everything - for now just set it before initial load
};

/* ========================================================================
   TWEAKS HOST BRIDGE
   ======================================================================== */

window.addEventListener('message', (e) => {
  const d = e.data || {};
  if (d.type === '__activate_edit_mode') tweaksPanel.classList.add('visible');
  if (d.type === '__deactivate_edit_mode') tweaksPanel.classList.remove('visible');
});
function persistTweaks(edits) {
  try {
    window.parent.postMessage({ type: '__edit_mode_set_keys', edits }, '*');
  } catch (e) {}
}
try { window.parent.postMessage({ type: '__edit_mode_available' }, '*'); } catch (e) {}

// Auto-select a showcase node after 2s so the right panel has content on first load
setTimeout(() => {
  if (state.selectedId == null) {
    const hub = graph.nodes.find(n => n.hub && n.region === 'hippo') || graph.nodes.find(n => n.hub);
    if (hub) selectNode(hub.id);
  }
}, 1800);
