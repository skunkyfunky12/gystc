# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the GYSTC Dashboard.

Build:  pyinstaller gystc.spec
Output: dist/GYSTC Dashboard/  (Windows)
        dist/GYSTC Dashboard.app/  (macOS)
"""

import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None
ROOT = Path(SPECPATH)
IS_MAC = sys.platform == 'darwin'


def _read_version() -> str:
    """The one version source: brain_mcp/__init__.py.

    The macOS bundle used to carry a hardcoded '1.0.0' while pyproject said
    1.4.3 and the package said 1.4.1 -- an artifact that told three different
    stories about what it was. Read the file rather than importing it, so it is
    unambiguous which copy is read on a machine that also has gystc installed.
    """
    text = (ROOT / 'brain_mcp' / '__init__.py').read_text(encoding='utf-8')
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise SystemExit('cannot read __version__ from brain_mcp/__init__.py')
    return match.group(1)


VERSION = _read_version()

# sentence-transformers loads its modules dynamically (modules.json names the
# classes), so static analysis finds none of them. Without this the bundle
# imports but every model load fails.
_st_datas, _st_binaries, _st_hidden = collect_all('sentence_transformers')

# assets/ is shipped whole (see datas below), so a model saved to assets/model
# by the release workflow travels with the app. That is what lets the packaged
# build embed with HF_HUB_OFFLINE=1 on a machine that never saw the model.

icon_file = ROOT / 'assets' / ('gystc-icon.icns' if IS_MAC else 'gystc-icon.ico')
if not icon_file.exists():
    icon_file = ROOT / 'assets' / 'gystc-icon.ico'

a = Analysis(
    [str(ROOT / 'main.py')],
    pathex=[str(ROOT)],
    binaries=_st_binaries,
    datas=[
        (str(ROOT / 'brain' / 'web'), 'brain/web'),
        (str(ROOT / 'brain' / 'shaders'), 'brain/shaders'),
        (str(ROOT / 'assets'), 'assets'),
        (str(ROOT / 'CLAUDE_TEMPLATE.md'), '.'),
    ] + _st_datas,
    hiddenimports=[
        'PyQt6.QtWebEngineWidgets',
        'PyQt6.QtWebEngineCore',
        'PyQt6.QtWebChannel',
        'PyQt6.QtNetwork',
        'scipy.spatial',
        'scipy.spatial._ckdtree',
        'numpy',
        'requests',
        'data',
        'data.loader',
        'data.vault_loader',
        'data.brain_layout',
        'data.regions',
        'brain',
        'brain.physics',
        'brain.web_widget',
        'brain.selfcheck',
        'integrations',
        'integrations.obsidian',
        'setup_wizard',
        # brain_mcp is imported in-process by the dashboard's HTTP API
        # (brain/web_widget.py: /api/config, /api/stats, /api/search, ...).
        # It must be bundled, NOT excluded.
        'brain_mcp',
        'brain_mcp.config',
        'brain_mcp.storage.database',
        'brain_mcp.storage.migrations',
        'brain_mcp.storage.file_lock',
        'brain_mcp.indexer.vector_store',
        'brain_mcp.indexer.embedder',
        'brain_mcp.indexer.bundled_model',
        'brain_mcp.indexer.chunker',
        'brain_mcp.indexer.scanner',
        'brain_mcp.indexer.pipeline',
        'brain_mcp.tools.retrieve',
        'brain_mcp.tools.recent',
    ] + _st_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'PIL',
        'IPython',
        'notebook',
        'pytest',
        'OpenGL',
        'brain.gl_widget',
        'brain.scene',
        'brain.camera',
        'brain.picking',
        'brain.window',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GYSTC Dashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(icon_file),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GYSTC Dashboard',
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name='GYSTC Dashboard.app',
        icon=str(icon_file),
        bundle_identifier='dev.gystc.dashboard',
        info_plist={
            'CFBundleDisplayName': 'GYSTC Dashboard',
            'CFBundleShortVersionString': VERSION,
            'CFBundleVersion': VERSION,
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'LSEnvironment': {
                'QTWEBENGINE_CHROMIUM_FLAGS': '--in-process-gpu',
            },
        },
    )
