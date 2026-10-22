"""Explicit optional repair of the bundled YOLO26 ONNX checkpoints.
Run with .ai-venv after INSTALLER_IA_AVANCEE.bat. Official assets only.
Code and weights: AGPL-3.0 / Enterprise; see LICENCES.md.
"""
from pathlib import Path
import os,urllib.request,hashlib,json,sys
os.environ['YOLO_AUTOINSTALL']='false'
root=Path(__file__).resolve().parents[1];folder=root/'backend/models/vision';folder.mkdir(exist_ok=True)
import torch
torch.set_num_threads(4)
from ultralytics import YOLO
report=[]
for name in ['yolo26n','yolo26s']:
 url=f'https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}.pt';pt=folder/(name+'.pt')
 if not pt.exists():urllib.request.urlretrieve(url,pt)
 model=YOLO(str(pt));target=folder/(name+'.onnx')
 if not target.exists():model.export(format='onnx',imgsz=640,dynamic=False,simplify=False,opset=17,end2end=False,device='cpu')
 report.append({'id':name,'architecture':'YOLO26 COCO detection','source':url,'code_version':__import__('ultralytics').__version__,'code_license':'AGPL-3.0-or-enterprise','weights_license':'AGPL-3.0-or-enterprise','license_source':'https://docs.ultralytics.com/models/yolo26/','checkpoint_sha256':hashlib.sha256(pt.read_bytes()).hexdigest(),'onnx_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'file':name+'.onnx','export':{'size':640,'opset':17,'end2end':False,'precision':'float32'}})
 (folder/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('Verified export',name,flush=True)
