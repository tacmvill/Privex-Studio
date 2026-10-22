"""Resolve one manual rectangle, reusing local models and existing native detections."""
from threading import Lock
import numpy as np
from videomask.persons import iou

_lock=Lock()
_detector=None


def choose(objects,box):
 x,y,w,h=box;ranked=[]
 for obj in objects:
  bx,by,bw,bh=obj['box'];intersection=max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by));area=max(1,bw*bh)
  if not x<=bx+bw/2<=x+w or not y<=by+bh/2<=y+h or intersection/area<.55 or obj.get('confidence',obj.get('detection_confidence',0))<.45:continue
  overlap=iou(tuple(box),tuple(obj['box']));ratio=area/max(1,w*h)
  if overlap<.2 or not .15<=ratio<=2:continue
  score=.75*overlap+.2*intersection/area+.05*obj.get('confidence',obj.get('detection_confidence',0));ranked.append((score,obj))
 ranked.sort(key=lambda v:v[0],reverse=True)
 if not ranked:return None
 if len(ranked)>1 and ranked[0][0]-ranked[1][0]<.1 and iou(tuple(ranked[0][1]['box']),tuple(ranked[1][1]['box']))<.6:return None
 return ranked[0][1]


def resolve(mid,seed,cfg):
 global _detector
 from app import Config,media
 from engine import COCO_NAMES,Detector
 from inspection import source_at,load_inspection
 from foreground_tracking import foreground,polygon
 m=media(mid);index=min(m['frames']-1,max(0,round(seed.time*m['fps'])));frame=source_at(m,index)
 box=[round(v*n/100) for v,n in zip(seed.box,[m['width'],m['height'],m['width'],m['height']])];found=None;provider='Suivi local du premier plan'
 with _lock:
  for ref in dict.fromkeys([cfg.selection_ref,cfg.inspection_ref]):
   if not ref:continue
   try:
    data=load_inspection(ref,mid);key=data.get('plan_ref','');rows=__import__('plan_store').frame(key,index) if len(key)==64 else None
    if rows is None:rows=next((r['objects'] for r in data['samples'] if r['frame']==index),[])
    found=choose(rows,box)
    if found:provider='Détection locale déjà enregistrée';break
   except (ValueError,OSError):pass
  if found is None:
   if _detector is None:_detector=Detector(Config(profile='rapide',scope='tous',heads=True,faces=True,classes=COCO_NAMES,confidence=.4))
   x,y,w,h=box;pad=round(max(w,h)*.3);xa=max(0,x-pad);ya=max(0,y-pad);crop=frame[ya:min(m['height'],y+h+pad),xa:min(m['width'],x+w+pad)]
   objects=[]
   for obj in _detector.detect(crop):
    obj=dict(obj);bx,by,bw,bh=obj['box'];obj['box']=[bx+xa,by+ya,bw,bh];objects.append(obj)
   found=choose(objects,box);provider=_detector.provider
  selected=found['box'] if found else box;mask,separated=foreground(frame,selected)
 result={'recognized':found is not None,'box':[v/n*100 for v,n in zip(selected,[m['width'],m['height'],m['width'],m['height']])],'category':(found.get('category') or found['class']) if found else 'custom','label':(found.get('category') or found['class']).capitalize() if found else seed.label,'foreground_separated':separated,'provider':provider,'confidence':float(found.get('confidence',found.get('detection_confidence',0))) if found else None}
 result['foreground_polygon']=[[[point[0]/m['width']*100,point[1]/m['height']*100] for point in outline] for outline in polygon(mask)] if separated else []
 return result
