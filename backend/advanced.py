"""Opt-in official Grounding DINO, SAM 2 and BoT-SORT + learned ReID. Local inference."""
from pathlib import Path
import os
# DirectML provides onnxruntime. Ultralytics must not install a competing wheel.
os.environ['YOLO_AUTOINSTALL']='false'
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent/'models/advanced'
_sam=None
def installed():
 runtimes=[ROOT.parents[2]/'.ai-venv/Lib/site-packages',ROOT.parents[2]/'.venv/Lib/site-packages']
 transformers=any((p/'torch').exists() and (p/'transformers').exists() for p in runtimes)
 reid=any((p/'torch').exists() and (p/'ultralytics').exists() and (p/'lap').exists() for p in runtimes)
 return {'Grounding DINO':transformers and (ROOT/'grounding-dino/model.safetensors').exists(),'SAM 2':transformers and (ROOT/'sam2/model.safetensors').exists(),'BoT-SORT + ReID':reid and (ROOT/'yolo26n-reid.onnx').exists()}
class GroundedDetector:
 def __init__(self,prompts):
  import torch
  from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
  torch.set_num_threads(4);self.torch=torch;self.prompts=prompts
  self.processor=AutoProcessor.from_pretrained(ROOT/'grounding-dino',local_files_only=True)
  self.model=AutoModelForZeroShotObjectDetection.from_pretrained(ROOT/'grounding-dino',local_files_only=True).eval()
 def detect(self,frame,threshold):
  from PIL import Image
  h,w=frame.shape[:2];text='. '.join(self.prompts)+'.'
  inputs=self.processor(images=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),text=text.lower(),return_tensors='pt')
  with self.torch.inference_mode():outputs=self.model(**inputs)
  results=self.processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=threshold,text_threshold=.25,target_sizes=[(h,w)])[0]
  rows=[]
  for box,score,label in zip(results['boxes'],results['scores'],results['text_labels']):
   x1,y1,x2,y2=box.tolist();name=label.strip()
   aliases={'person':'personne','car':'voiture','bicycle':'vélo','dog':'chien','cat':'chat','truck':'camion','chair':'chaise','airplane':'avion','bus':'bus','motorcycle':'moto','clock':'horloge','bottle':'bouteille','scooter':'trottinette'}
   import re
   category=next((fr for en,fr in aliases.items() if re.search(r'\b'+en+r'\b',name.casefold())),name)
   from engine import CLASS_IDS
   import zlib
   cls=CLASS_IDS.get(category,1400+(zlib.crc32(name.casefold().encode('utf-8'))&65535))
   rows.append({'box':[max(0,round(x1)),max(0,round(y1)),max(1,round(x2-x1)),max(1,round(y2-y1))],'class':name,'category':category,'class_id':cls,'confidence':float(score),'review_required':True,'method':'Grounding DINO'})
  return rows
class Segmenter:
 def __init__(self):
  import torch
  from transformers import Sam2Processor,Sam2Model
  torch.set_num_threads(4);self.torch=torch
  self.processor=Sam2Processor.from_pretrained(ROOT/'sam2',local_files_only=True)
  self.model=Sam2Model.from_pretrained(ROOT/'sam2',local_files_only=True).eval()
 def annotate(self,frame,objects):
  if not objects:return objects
  from PIL import Image
  boxes=[[[o['box'][0],o['box'][1],o['box'][0]+o['box'][2],o['box'][1]+o['box'][3]] for o in objects]]
  inputs=self.processor(images=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),input_boxes=boxes,return_tensors='pt')
  with self.torch.inference_mode():outputs=self.model(**inputs,multimask_output=False)
  masks=self.processor.post_process_masks(outputs.pred_masks.cpu(),inputs['original_sizes'])[0]
  for obj,mask in zip(objects,masks):
   binary=mask[0].numpy().astype(np.uint8)*255;contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   obj['polygons']=[cv2.approxPolyDP(c,1.5,True).reshape(-1,2).tolist() for c in contours if cv2.contourArea(c)>12];obj['segmentation']='SAM 2'
  return objects
def annotate(frame,objects):
 global _sam
 if _sam is None:_sam=Segmenter()
 return _sam.annotate(frame,objects)
class AppearanceTracker:
 def __init__(self,fps,namespace=""):
  self.namespace=namespace
  from ultralytics.trackers.bot_sort import BOTSORT
  from types import SimpleNamespace
  import torch
  torch.set_num_threads(4)
  args=SimpleNamespace(track_high_thresh=.3,track_low_thresh=.1,new_track_thresh=.3,track_buffer=max(30,round(fps*8)),match_thresh=.8,fuse_score=True,gmc_method='sparseOptFlow',proximity_thresh=.3,appearance_thresh=.7,with_reid=True,model=str(ROOT/'yolo26n-reid.onnx'),device='cpu')
  self.tracker=BOTSORT(args);self.colors={}
  if self.tracker.encoder is not None:self.tracker.encoder=__import__('reid_cache').StableEncoder(self.tracker.encoder,max_age=max(1,round(fps*.1)))
 def update(self,objects,frame):
  from ultralytics.engine.results import Boxes
  from engine import PALETTE
  if self.tracker.encoder is not None:self.tracker.encoder.frame_index=self.tracker.frame_id+1
  data=np.array([[o['box'][0],o['box'][1],o['box'][0]+o['box'][2],o['box'][1]+o['box'][3],o['confidence'],o['class_id']] for o in objects],np.float32).reshape(-1,6)
  tracked=self.tracker.update(Boxes(data,frame.shape[:2]),img=frame);result=[]
  for row in tracked:
   x1,y1,x2,y2,track_id,confidence,cls,index=row[:8];original=objects[int(index)];ident=f'a{self.namespace}-bot{int(track_id)}'
   if ident not in self.colors:self.colors[ident]=PALETTE[len(self.colors)%len(PALETTE)]
   result.append(dict(original,id=ident,box=[round(x1),round(y1),round(x2-x1),round(y2-y1)],confidence=float(confidence),color=self.colors[ident]))
  return result
