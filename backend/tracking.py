"""Subject anchors, detector-assisted CSRT and explicit review of missing intervals."""
import cv2,numpy as np
from videomask.custom_tracking import ObjectFollower
from videomask.persons import iou

class ReviewRequired(Exception):
 def __init__(self,plan,gaps):self.plan=plan;self.gaps=gaps;super().__init__('Suivi à vérifier avant export.')

def intervals(indices):
 if not indices:return []
 result=[];start=last=indices[0]
 for n in indices[1:]:
  if n!=last+1:result.append((start,last));start=n
  last=n
 return [*result,(start,last)]

def gap_confirmed(gap,confirmed,fps):
 """A confirmed absence still covers shorter gaps after a corrected tracking pass."""
 if gap['id'] in confirmed:return True
 start=round(gap['start']*fps);end=round(gap['end']*fps)-1;spans=[]
 key=lambda v:v[3:] if v.startswith('m-da') else v
 aliases={key(gap['subject_id']),*[key(v) for v in gap.get('reference_ids',[])]}
 for value in confirmed:
  try:
   subject,a,b,_=value.rsplit(':',3);a=int(a);b=int(b)
  except (ValueError,AttributeError):continue
  if key(subject) in aliases and a<=b:spans.append((a,b))
 cursor=start
 for a,b in sorted(spans):
  if b<cursor:continue
  if a>cursor:return False
  cursor=max(cursor,b+1)
  if cursor>end:return True
 return False

def track_subjects(m,cfg,frames,job,frame_range=None,presence=None):
 from inspection import source_at
 from app import _clip_seed,Anchor
 from temporal_intelligence import HeadBodyBridge,IdentityMemory,IndependentVerifier,evaluate_subject,compatible,category
 from types import SimpleNamespace
 from performance import phase,tick,timed
 pending_evaluations=[]
 first,last=frame_range or (0,m['frames']);cuts=presence.cuts if presence else [];gaps=[];trajectories=[]
 active=[s.model_copy(deep=True) for s in cfg.custom if not s.hidden];raw=[list(row) for row in frames]
 if m['type']=='image':
  for seed in active:
   box=_clip_seed(tuple(round(v*n/100) for v,n in zip(seed.box,[m['width'],m['height'],m['width'],m['height']])),m)
   frames[0].append({'id':seed.id,'class':seed.label,'category':seed.category,'color':seed.color,'box':list(box)})
  return []
 if cfg.scope=='tous':
  phase('Tracking',1,'passes')
  job['automatic_reidentifications']=__import__('temporal_intelligence').reconcile_automatic_tracks(m,frames,cuts,job,presence.bodies if presence else None)
  job['identity_groups']=__import__('identity_groups').consolidate(frames)
  __import__('retrospective_tracking').recover(m,frames,cuts,job,presence,first,last);raw=[list(row) for row in frames]
 verifier=IndependentVerifier(cfg);verifier.body_evidence=presence.bodies if presence else {}
 def normalized_box(box):return [v/n*100 for v,n in zip(box,[m['width'],m['height'],m['width'],m['height']])]
 # Multiple references to the same real detector trajectory form one subject.
 grouped={};claimed_automatic=set()
 for seed in active:
  index=min(m['frames']-1,round(seed.time*m['fps']));box=tuple(v*n/100 for v,n in zip(seed.box,[m['width'],m['height'],m['width'],m['height']]))
  matches=[o for o in raw[index] if compatible(category(o),(seed.category or seed.label).casefold()) and iou(tuple(o['box']),box)>.6]
  known=max(matches,key=lambda o:iou(tuple(o['box']),box))['id'] if matches and (seed.id.startswith('m-da') or seed.smart) else None
  if known:claimed_automatic.add(known)
  group=('detected',known) if known else ('manual',seed.id)
  if group not in grouped:
   seed.reference_ids=list(set([*seed.reference_ids,*([seed.id[3:]] if seed.id.startswith('m-da') else [])]));grouped[group]=seed
  else:
   previous=grouped[group];previous.anchors += [Anchor(time=seed.time,box=seed.box),*seed.anchors];previous.reference_ids+=seed.reference_ids+[seed.id[3:] if seed.id.startswith('m-da') else seed.id]
 active=list(grouped.values())
 phase('Tracking',max(1,len(active)),'sujets')
 for number,seed in enumerate(active):
  job['phase']=f'Identité, visibilité et masquage · {seed.label}';observations={};anchors={};cat=(seed.category or seed.label).casefold()
  for row in frames:row[:]=[o for o in row if o['id']!=seed.id]
  for a in [seed,*seed.anchors]:
   index=min(m['frames']-1,round(a.time*m['fps']));anchors[index]=_clip_seed(tuple(round(v*n/100) for v,n in zip(a.box,[m['width'],m['height'],m['width'],m['height']])),m)
  ordered=sorted(anchors)
  for anchor,box in anchors.items():
   frame0=source_at(m,anchor)
   objects=[o for o in raw[anchor] if compatible(category(o),cat) and iou(tuple(o['box']),box)>.3]
   # References can lie outside a retained timeline clip; verify that anchor once.
   if not objects and presence and seed.category not in {'','custom'}:
    objects=[o for o in presence.detector.detect(frame0) if compatible(category(o),cat) and iou(tuple(o['box']),box)>.3]
   if presence and cat in {'tête','visage'} and not presence.bodies.get(anchor):presence.record(anchor,presence.detector.detect(frame0))
   detection=max(objects,key=lambda o:iou(tuple(o['box']),box)) if objects else {}
   base={'box':list(box),'visible':True,'detected':bool(detection),'detection_confidence':.55 if detection.get('confidence_estimated') else detection.get('confidence',0),'confidence_estimated':detection.get('confidence_estimated',False),'tracking_confidence':1.,'appearance_confidence':1.,'existence_verified':detection.get('existence_verified',False),'verification_uncertain':detection.get('verification_uncertain',False)}
   observations[anchor]=base
   for direction in [1,-1]:
    if direction==1 and anchor>=last-1 or direction==-1 and anchor<=first:continue
    follower_type=__import__('foreground_tracking').ForegroundFollower if seed.smart else ObjectFollower
    follower=follower_type(frame0,box);memory=IdentityMemory(frame0,box,index=anchor,person=cat=='personne');memory.last_index=anchor
    if seed.smart:observations[anchor].update(polygons=follower.polygons,foreground_separated=follower.separated)
    bridge=HeadBodyBridge(frame0,box,presence.bodies.get(anchor,[]),anchor) if presence and cat in {'tête','visage'} else None
    cap=cv2.VideoCapture(str(m['path']));index=anchor+direction;lost=False
    if direction==1:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    boundary=next((a for a in (ordered if direction>0 else ordered[::-1]) if (a-anchor)*direction>0),last if direction>0 else first-1)
    boundary=min(boundary,last) if direction>0 else max(boundary,first-1)
    try:
     while 0<=index<m['frames'] and (boundary-index)*direction>0:
      if job['cancel']:raise InterruptedError('Suivi annulé.')
      if (index if direction>0 else index+1) in cuts:break
      if direction<0:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
      frame=__import__('video_io').frame_at(m,index,cap)
      candidates=[o for o in [*raw[index],*(getattr(presence,'subjects',{}).get(index,[]) if seed.smart and presence else [])] if compatible(category(o),cat) and o.get('detected',True)]
      best,ambiguous,appearance=memory.choose(frame,candidates,index,m['fps'],lost)
      if bridge:
       assisted,body_confidence=bridge.choose(frame,candidates,presence.bodies.get(index,[]),index,m['fps'],lost)
       if best is None and assisted is not None:best=assisted;appearance=body_confidence;ambiguous=False
       elif best is not None and assisted is not None and iou(tuple(best['box']),tuple(assisted['box']))<.3:best=None;ambiguous=True
      retry=seed.smart and abs(index-anchor)%max(1,round(m['fps']*.2))==0
      tracked=None if lost and not retry or best is not None and not ambiguous else follower.update(frame)
      quality=getattr(follower,'confidence',0.) if tracked is not None else 0.
      was_lost=lost
      if best is not None and not ambiguous:
       tracked=tuple(best['box']);quality=1.
       if lost or index%5==0 or iou(tuple(follower.box),tracked)<.6:follower=follower_type(frame,tracked)
       else:follower.box=tracked;follower.misses=0
       lost=False
      elif tracked is None or follower.misses or quality<.6:
       lost=True;tracked=None
      if tracked is not None:
       lost=False
       evidence={'box':list(tracked),'visible':True,'detected':best is not None,'detection_confidence':(.55 if best.get('confidence_estimated') else best.get('confidence',0)) if best else 0.,'confidence_estimated':best.get('confidence_estimated',False) if best else False,'tracking_confidence':quality,'appearance_confidence':appearance,'identity_ambiguous':ambiguous,'reidentified':was_lost and best is not None}
       if seed.smart:evidence.update(polygons=follower.polygons,foreground_separated=follower.separated)
       old=observations.get(index)
       if old is None or quality>old['tracking_confidence']:observations[index]=evidence
       if not ambiguous:memory.remember(frame,tracked,index)
      index+=direction
    finally:cap.release()
  proposals={i:[(abs(i-round(seed.time*m['fps'])),tuple(o['box']),not o.get('visible'))] for i,o in observations.items()}
  absent=presence.absent_frames(seed,anchors,proposals)|presence.explained_edges(seed,observations,raw) if presence else set()
  pending_evaluations.append((seed,observations,anchors,absent,first,last));tick(number+1)
 # Global discovery evaluates each automatic trajectory independently. Candidates
 # remain discoverable even when they are not currently allowed to produce a mask.
 if cfg.scope=='tous':
  occurrences={}
  for index,row in enumerate(raw):
   if not first<=index<last:continue
   for obj in row:
    if obj['id'].startswith('a') and obj['id'] not in claimed_automatic:occurrences.setdefault(obj['id'],{})[index]=dict(obj)
  for row in frames:row[:]=[o for o in row if o['id'] not in occurrences and o['id'] not in claimed_automatic]
  for number,(ident,observations) in enumerate(occurrences.items()):
   at=min(observations);sample=observations[at];cat=category(sample)
   seed=SimpleNamespace(id=ident,label=sample['class'],category=cat,color=sample['color'],time=at/m['fps'],box=normalized_box(sample['box']),reference_ids=sample.get('reference_ids',[]))
   for obj in observations.values():
    obj.setdefault('tracking_confidence',1.);obj.setdefault('visible',True);obj.setdefault('detection_confidence',obj.get('confidence',0))
   anchors={i:tuple(o['box']) for i,o in observations.items() if o.get('detected',True)}
   proposals={i:[(0,tuple(o['box']),not o['visible'])] for i,o in observations.items()}
   absent=presence.absent_frames(seed,anchors,proposals)|presence.explained_edges(seed,observations,raw) if presence else set()
   # Discovery inventories an observed lifetime; selected export independently validates the full retained range.
   begin,end=(min(observations),max(observations)+1) if cfg.catalog_only else (first,last)
   pending_evaluations.append((seed,observations,anchors,absent,begin,end))
   tick(number+1,len(occurrences))
 phase('Vérification temporelle',len(pending_evaluations),'sujets')
 for number,(seed,observations,anchors,absent,begin,end) in enumerate(pending_evaluations,1):
  with timed('verification/temporal'):rows,issues,trajectory=evaluate_subject(m,seed,observations,anchors,absent,cuts,begin,end,verifier,job)
  if cfg.catalog_only:trajectory['catalogue_only']=True
  for i,obj in rows:frames[i].append(obj)
  gaps+=issues;trajectories.append(trajectory);tick(number)
 # Competing selections may not silently claim the same subject at a crossing.
 conflicts={}
 for i in range(first,last):
  row=[o for o in frames[i] if o['id'] in {s.id for s in active} and o.get('visible')]
  for n,a in enumerate(row):
   for b in row[n+1:]:
    if compatible(category(a),category(b)) and iou(tuple(a['box']),tuple(b['box']))>.75:
     for obj in [a,b]:obj.update(mask_allowed=False,presence='doubt',identity_ambiguous=True);conflicts.setdefault(obj['id'],[]).append(i)
 for ident,indices in conflicts.items():
  seed=next(s for s in active if s.id==ident)
  for a,b in intervals(sorted(set(indices))):gaps.append({'id':f'{ident}:{a}:{b}:Identité à confirmer','subject_id':ident,'label':seed.label,'start':a/m['fps'],'end':(b+1)/m['fps'],'reason':'Deux sujets revendiquent la même position','kind':'identity','reference_time':seed.time,'reference_box':seed.box,'category':seed.category,'color':seed.color})
  for trajectory in trajectories:
   if trajectory['id']==ident:
    values=[(i,'doubt' if i in set(indices) else span['state']) for span in trajectory['intervals'] for i in range(span['first'],span['last']+1)]
    trajectory['intervals']=__import__('temporal_intelligence').spans(values,m['fps'])
 job['trajectories']=trajectories
 return gaps
