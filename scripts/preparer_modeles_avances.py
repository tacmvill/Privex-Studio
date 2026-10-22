from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
import urllib.request,json
root=Path(__file__).resolve().parents[1]/'backend/models/advanced';root.mkdir(exist_ok=True)
def download(spec):
 name,repo=spec
 path=snapshot_download(repo,local_dir=root/name,allow_patterns=['*.json','*.txt','*.safetensors','*.model'],cache_dir=root/'hf-cache')
 print(name,'ready',flush=True);return name
with ThreadPoolExecutor(max_workers=2) as pool:
 for result in pool.map(download,[('grounding-dino','IDEA-Research/grounding-dino-tiny'),('sam2','facebook/sam2.1-hiera-tiny')]):pass
urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-reid.onnx',root/'yolo26n-reid.onnx')
(root/'ready.json').write_text(json.dumps({'grounding-dino':True,'sam2':True,'reid':True}))
print('All advanced models ready',flush=True)
