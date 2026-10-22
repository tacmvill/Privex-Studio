"""Automatic orchestration over the tested V9 adapters; no detector selection in the UI."""
from videomask.persons import iou

def selected_categories(cfg):
 result=set(cfg.classes)|set(cfg.extended)
 if cfg.heads:result.add('tête')
 elif cfg.faces:result.add('visage')
 if cfg.plates:result.add('plaque')
 if cfg.ocr:result|={'texte','heure lisible','texte de panneau'}
 return result

def requested(obj,cfg):
 cat=obj.get('category',obj.get('class','')).casefold()
 return cat in selected_categories(cfg) or cfg.brands and cat.startswith('marque ') or bool(cfg.grounding_prompts) and obj.get('model_source')=='Grounding DINO'

class PrivexVisionEngine:
 def __init__(self,cfg):
  from engine import LegacyDetector
  self.cfg=cfg;self.adapter=LegacyDetector(cfg);self.provider=self.adapter.provider;self.manager=self.adapter.manager;self.verifier=None;self.regions={}
 def detect(self,frame):
  rows=self.adapter.detect(frame,face_quality='base' if self.cfg.profile!='rapide' else 'standard')
  output=[]
  for obj in sorted(rows,key=lambda o:o['confidence'],reverse=True):
   obj=dict(obj);cat=obj.get('category',obj['class']).casefold();obj['category']=cat
   obj['model_source']=obj.get('model_source') or ('HollywoodHeads' if cat=='tête' else 'YuNet/SCRFD' if cat=='visage' else 'Grounding DINO' if obj.get('method')=='Grounding DINO' else 'YOLOE/OCR' if cat not in __import__('engine').CLASS_IDS else self.adapter.yolo.path.stem if self.adapter.yolo else 'V9')
   obj['model_confidences']={obj['model_source']:obj['confidence']};obj['auxiliary']=cat in self.cfg.evidence_classes and not requested(obj,self.cfg)
   if not requested(obj,self.cfg) and not obj['auxiliary']:
    if cat in {'voiture','bus','camion'} and set(self.cfg.classes)&{'voiture','bus','camion'}:obj.update(vehicle_continuation=True,continuation_only=True)
    else:continue
   previous=next((p for p in output if p['category']==cat and iou(tuple(p['box']),tuple(obj['box']))>.55),None)
   if previous:previous['model_confidences'].update(obj['model_confidences']);continue
   output.append(obj)
  for head in [o for o in output if o['category'] in {'tête','visage'}]:
   x,y,w,h=head['box'];body=next((b for b in output if b['category']=='personne' and b['box'][0]<=x+w/2<=b['box'][0]+b['box'][2] and b['box'][1]<=y+h/2<=b['box'][1]+b['box'][3]*.6),None)
   if body:head['body_evidence']={'box':body['box'],'confidence':body['confidence'],'model_source':body['model_source']}
  # Regional corroboration is deferred to the temporal pass: at most three
  # representative native crops per uncertain trajectory, not every weak proposal.
  for person in [o for o in output if o['category']=='personne' and not o['auxiliary']]:
   if person['confidence']<.6:person.update(review_required=True,verification_pending=True)
  self.provider=self.adapter.provider
  return __import__('proposals').unique(output)
 def report(self):return {**self.manager.report(),'actual_provider':self.provider}

 def deep(self,frame,regions,full=False,thorough=False):
  """Scheduled discovery tiles and native difficult regions; no nested tiling."""
  h,w=frame.shape[:2];result=[]
  if full:
   cw=round(w*.62);ch=round(h*.62);regions=[(x,y,cw,ch) for x,y in [(0,0),(w-cw,0),(0,h-ch),(w-cw,h-ch)]]+regions
  for region_number,box in enumerate(regions):
   x,y,bw,bh=map(round,box);x=max(0,x);y=max(0,y);crop=frame[y:min(h,y+bh),x:min(w,x+bw)]
   if not crop.size:continue
   for obj in self.adapter.detect(crop,face_quality=('precise' if thorough else 'regional') if not full else 'base',human_only=True,skip_body=(not thorough and not (full and region_number<4) and bool(self.adapter.head or self.adapter.face))):
    if obj['class'] not in {'personne','tête','visage'}:continue
    obj=dict(obj);bx,by,bw,bh=obj['box'];obj['box']=[bx+x,by+y,bw,bh];obj['verification_pass']='targeted_native';obj['category']=obj.get('category',obj['class']);obj['model_source']=obj.get('model_source') or ('YuNet/SCRFD' if obj['class']=='visage' else self.adapter.yolo.path.stem if self.adapter.yolo else 'V9');obj['model_confidences']={obj['model_source']:obj['confidence']};obj['auxiliary']=obj['category'] in self.cfg.evidence_classes and not requested(obj,self.cfg)
    if requested(obj,self.cfg) or obj['auxiliary']:result.append(obj)
  return result
