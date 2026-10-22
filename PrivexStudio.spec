from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

root = Path(SPECPATH)
datas = [(str(p), str(p.relative_to(root).parent)) for p in (root/'backend').rglob('*')
         if p.is_file() and '__pycache__' not in p.parts and 'data' not in p.relative_to(root/'backend').parts
         and (p.suffix == '.py' or ('models' in p.parts and p.suffix in {'.onnx','.json'}))]
datas += [(str(root/'frontend/dist'), 'frontend/dist'),
         (str(root/'frontend/public/fonts'), 'frontend/public/fonts'),
         (str(root/'docs'), 'docs')]
binaries = []
hidden = [p.stem for p in (root/'backend').glob('*.py')]
hidden += ['uvicorn.logging','uvicorn.loops.auto','uvicorn.protocols.http.auto',
           'uvicorn.protocols.http.h11_impl','uvicorn.protocols.websockets.auto',
           'uvicorn.lifespan.on','webview.platforms.edgechromium','webview.platforms.winforms',
           'webview.platforms.cef','pythonnet','clr_loader']
for package in ['onnxruntime','rapidocr_onnxruntime','imageio_ffmpeg','trackers']:
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hidden += h
datas += collect_data_files('webview')
a = Analysis([str(root/'launch.py')], pathex=[str(root),str(root/'backend')],
    binaries=binaries, datas=datas, hiddenimports=hidden,
    excludes=['torch','torchvision','transformers','ultralytics','tensorflow','jax',
              'IPython','pytest','tkinter','PyQt5','PyQt6','PySide2','PySide6','cefpython3'],
    noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='PrivexStudio',
    console=False, icon=str(root/'branding/PrivexStudio.ico'))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='PrivexStudio')
