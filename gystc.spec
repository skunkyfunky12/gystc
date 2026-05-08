# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the GYSTC Dashboard.

Build:  pyinstaller gystc.spec
Output: dist/GYSTC Dashboard/  (Windows)
        dist/GYSTC Dashboard.app/  (macOS)
"""

import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH)
IS_MAC = sys.platform == 'darwin'

icon_file = ROOT / 'assets' / ('gystc-icon.icns' if IS_MAC else 'gystc-icon.ico')
if not icon_file.exists():
    icon_file = ROOT / 'assets' / 'gystc-icon.ico'

a = Analysis(
    [str(ROOT / 'main.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / 'brain' / 'web'), 'brain/web'),
        (str(ROOT / 'brain' / 'shaders'), 'brain/shaders'),
        (str(ROOT / 'assets'), 'assets'),
        (str(ROOT / 'CLAUDE_TEMPLATE.md'), '.'),
    ],
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
        'brain.gl_widget',
        'brain.scene',
        'brain.camera',
        'brain.picking',
        'brain.window',
        'integrations',
        'integrations.obsidian',
        'setup_wizard',
    ],
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
        'brain_mcp',
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
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'LSEnvironment': {
                'QTWEBENGINE_CHROMIUM_FLAGS': '--in-process-gpu',
            },
        },
    )
