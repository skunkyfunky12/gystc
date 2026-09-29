// Editable defaults for the tweaks panel -- a host may rewrite this block.
//
// A plain script loaded before brain-app.js rather than an inline <script>:
// inline code would need 'unsafe-inline' in the page's script-src, and that
// would also let any injected onerror= attribute run next to the API token.
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "palette": "graphite",
  "glow": 1.25,
  "stars": 8000,
  "nodeSize": 2.5,
  "pulseSpeed": 0.7,
  "bloom": 0.55,
  "bloomRadius": 0.7
}/*EDITMODE-END*/;
