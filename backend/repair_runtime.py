"""Explicit repair of overlapping ONNX Runtime distributions on Windows."""
import sys,os,subprocess,importlib.metadata as metadata

def main():
 installed={d.metadata['Name'].lower() for d in metadata.distributions()}
 if os.name!='nt':return
 if 'onnxruntime' in installed or 'onnxruntime-directml' not in installed:
  if '--repair' not in sys.argv:raise SystemExit('ONNX Runtime CPU/DirectML conflict. Close Privex, run REPARER_ACCELERATION.bat.')
  if 'onnxruntime' in installed:subprocess.run([sys.executable,'-m','pip','uninstall','-y','onnxruntime'],check=True)
  subprocess.run([sys.executable,'-m','pip','install','--force-reinstall','--no-deps','onnxruntime-directml==1.24.4'],check=True)
 # Verify in a fresh interpreter so a previously imported DLL cannot hide the result.
 subprocess.run([sys.executable,'-c',"import onnxruntime as o; print(o.__version__,o.get_available_providers()); assert 'DmlExecutionProvider' in o.get_available_providers(), 'DirectML unavailable; CPU fallback remains possible.'"],check=True)
if __name__=='__main__':main()
