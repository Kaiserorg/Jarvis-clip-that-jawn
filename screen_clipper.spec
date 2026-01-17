# PyInstaller spec file for screen_clipper
# This is an optional spec to allow reproducible builds with Tree('model') included.

# -*- mode: python -*-
import os
from PyInstaller.utils.hooks import Tree

block_cipher = None

datas = Tree('model', prefix='model') if os.path.isdir('model') else []

a = Analysis(['screen_clipper.py'],
             pathex=['.'],
             binaries=[],
             datas=datas,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='ScreenClipper',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)

coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name='ScreenClipper')
