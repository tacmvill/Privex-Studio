"""Prepare reusable silhouettes from sparse segmentation keyframes, never new subjects."""
import json,threading,cv2
import os,time
from contextlib import contextmanager
import numpy as np
from videomask.persons import iou

_lock=threading.Lock()

@contextmanager
def disk_lock(path):
 with path.open('a+b') as stream:
  stream.seek(0,2)
  if stream.tell()==0:stream.write(b'0');stream.flush()
  if os.name=='nt':
   import msvcrt
   for attempt in range(400):
    try:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1);break
    except OSError:
     if attempt==399:raise ValueError('Les contours sont déjà en préparation. Réessayez dans un instant.')
     time.sleep(.05)
  try:yield
  finally:
   if os.name=='nt':stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)

def prepare(mid,cfg,job=None,recover_onsets=False):
 from app import DATA,media
 from inspection import load_inspection,library_seeds,source_at,subject_key
 from engine import CLASS_IDS
 from foreground_tracking import foreground,polygon
 from performance import timed,phase,tick
 m=media(mid)
 if m['type']!='video':return
 wanted={subject_key(s.id) for s in library_seeds(cfg) if (s.style.shape if s.style else cfg.shape)!='rectangle' or (s.style.effect if s.style else cfg.effect)=='pixels_noirs'}
 try:data=load_inspection(cfg.selection_ref or cfg.inspection_ref,mid)
 except ValueError:return
 key=data.get('plan_ref','')
 if len(key)!=64:return
 with _lock,disk_lock(DATA/(key+'.contours.lock')):
  path=DATA/(key+'.plan.json')
  if not path.exists():return
  plan=json.loads(path.read_text(encoding='utf-8'));tracks={}
  migrated=plan.get('render_continuity_version')!=3
  if plan.get('entry_guard_version')!=2:
   __import__('temporal_entry').extend(m,plan['frames'],plan.get('scene_cuts',[]),coverage=plan.get('analysis',{}).get('coverage'))
   plan['entry_guard_version']=2;migrated=True
  if recover_onsets:
   previous=plan.get('onset_checked_ids',[]) if plan.get('visible_onset_revision')==1 else []
   stats=__import__('visible_onsets').recover(m,plan['frames'],plan.get('scene_cuts',[]),job if job is not None else {'cancel':False},cfg,plan.get('analysis',{}).get('coverage'),previous)
   if stats['checked_ids']:
    plan['onset_checked_ids']=sorted(set([*previous,*stats['checked_ids']]));plan['visible_onset_revision']=1;plan['visible_onsets']=stats;migrated=True
  __import__('render_continuity').repair(plan)
  if plan.get('head_envelope_revision')!=1:
   __import__('head_envelopes').repair(plan,m);migrated=True
  if plan.get('appearance_padding_revision')!=1:
   __import__('appearance_padding').prepare(plan,m,cfg);migrated=True
  if plan.get('linear_tracking_revision')!=1:
   __import__('linear_tracking').prepare(plan,m['fps']);migrated=True
  if migrated:__import__('plan_store').save(key,plan)
  if not wanted:return
  for i,row in enumerate(plan['frames']):
   for obj in row:
    if not obj.get('visible',True) or obj.get('privacy_coverage'):continue
    if wanted.intersection({subject_key(obj['id']),*[subject_key(a) for a in obj.get('reference_ids',[])]}):tracks.setdefault(obj['id'],[]).append((i,obj))
  tracks={k:rows for k,rows in tracks.items() if any(not o.get('polygons') and not o.get('contour_attempted') for _,o in rows)}
  if not tracks:return
  if job is not None:job['phase']='Préparation des contours du montage'
  period=max(1,round(m['fps']*.5));requests={}
  for ident,rows in tracks.items():
   samples=[(i,o) for i,o in rows if not o.get('polygons') and i%period==0]
   if not samples:samples=[max(rows,key=lambda row:row[1].get('detection_confidence',0))]
   if rows[0][1].get('category',rows[0][1]['class']) in {'tête','visage'}:samples=[max(samples,key=lambda row:row[1].get('detection_confidence',0))]
   for i,o in samples:requests.setdefault(i,[]).append(o)
  shapes={};engine=__import__('engine');calls=0
  phase('Préparation des contours',len(requests),'images')
  for ordinal,(i,objects) in enumerate(sorted(requests.items()),1):
   if job is not None and job.get('cancel'):raise InterruptedError('Contours annulés.')
   frame=source_at(m,i);classes={CLASS_IDS.get(o.get('category',o['class']),-1) for o in objects};instances=[]
   if any(c>=0 for c in classes):
    if engine._segment is None:engine._segment=__import__('videomask.persons',fromlist=['PersonSegmentationONNX']).PersonSegmentationONNX()
    with timed('segmentation/contour_keyframes'):instances=engine._segment.predict(frame,classes={c for c in classes if c>=0})
    calls+=1
   for obj in objects:
    class_id=CLASS_IDS.get(obj.get('category',obj['class']),-1);matching=[p for p in instances if p.class_id==class_id and iou(tuple(p.box),tuple(obj['box']))>.25]
    if matching:
     mask=np.uint8(max(matching,key=lambda p:iou(tuple(p.box),tuple(obj['box']))).mask>0)*255
     if obj.get('category',obj['class'])=='personne':mask=__import__('body_contour').protect(mask,obj['box'])
     outline=polygon(mask)
    else:
     mask,separated=foreground(frame,obj['box']);outline=polygon(mask) if separated else []
     if not outline and obj.get('category',obj['class']) in {'tête','visage'}:
      x,y,w,h=map(round,obj['box']);mask=np.zeros(frame.shape[:2],np.uint8);cv2.ellipse(mask,(x+w//2,y+h//2),(max(1,w//2),max(1,h//2)),0,0,360,255,-1);outline=polygon(mask)
    if outline:shapes[(obj['id'],i)]=(obj['box'],outline)
   tick(ordinal)
  changed=False
  for ident,rows in tracks.items():
   previous=next((shapes[(ident,i)] for i,_ in rows if (ident,i) in shapes),None)
   for i,obj in rows:
    if i in plan.get('scene_cuts',[]):previous=None
    previous=shapes.get((ident,i),previous)
    if obj.get('polygons'):continue
    obj['contour_attempted']=True;changed=True
    if previous is None:continue
    old,outline=previous;x,y,w,h=obj['box'];ox,oy,ow,oh=old
    obj['polygons']=[[[x+(px-ox)*w/max(1,ow),y+(py-oy)*h/max(1,oh)] for px,py in poly] for poly in outline];obj['contour_source']='segmentation_keyframe';changed=True
  if changed:
   __import__('plan_store').save(key,plan)
   from app import PLANS
   for cached_key in list(PLANS):
    if PLANS[cached_key].get('analysis',{}).get('media_id')==mid:PLANS.pop(cached_key,None)
  if job is not None:job['contour_keyframe_inferences']=calls
