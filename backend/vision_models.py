"""Local model registry: declared checkpoints, hardware policy and fallback reporting."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parent
class ModelManager:
 def __init__(self,profile):
  import onnxruntime as ort
  try:
   import psutil
   memory=psutil.virtual_memory();self.ram_gb=round(memory.total/1024**3,1);self.available_gb=round(memory.available/1024**3,1)
  except ImportError:self.ram_gb=None;self.available_gb=None
  self.profile=profile;self.providers=ort.get_available_providers();self.warnings=[];self.models=[]
 def object_path(self):
  name='yolo26s' if self.profile!='rapide' and (self.available_gb is None or self.available_gb>1.25) else 'yolo26n'
  if self.profile!='rapide' and name=='yolo26n':self.warnings.append('Mémoire disponible limitée : modèle compact, avec les vérifications du mode Fidèle.')
  preferred=ROOT/'models/vision'/f'{name}.onnx'
  if not preferred.is_file():
   self.warnings.append(f'{name} absent : repli sur YOLO11n.');preferred=ROOT/'models/yolo11n.onnx'
  if not preferred.is_file():raise FileNotFoundError('Aucun modèle local de personnes/objets utilisable.')
  self.models.append(preferred.stem);return preferred
 def load_objects(self):
  from videomask.core import YoloDetector
  path=self.object_path()
  try:return YoloDetector(path)
  except Exception as exc:
   fallback=ROOT/'models/yolo11n.onnx'
   if path==fallback:raise RuntimeError('Le modèle local de personnes ne peut pas être chargé.') from exc
   self.warnings.append(f'{path.stem} incompatible avec ce runtime : repli YOLO11n.');self.models.remove(path.stem);self.models.append('yolo11n')
   return YoloDetector(fallback)
 def report(self):return {'engine':'Privex Vision Engine','mode':'Rapide' if self.profile=='rapide' else 'Fidèle','models':self.models,'available_providers':self.providers,'ram_gb':self.ram_gb,'available_ram_gb':self.available_gb,'warnings':self.warnings}

def inventory():
 try:manifest=json.loads((ROOT/'models/vision/manifest.json').read_text(encoding='utf-8'))
 except (OSError,ValueError):manifest=[]
 return {'models':[{**m,'installed':(ROOT/'models/vision'/m['file']).is_file()} for m in manifest],'hardware':ModelManager('rapide').report()}
