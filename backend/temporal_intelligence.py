"""Privex Temporal Intelligence: existence, identity and visible mask are separate facts.

Uncertainty never constitutes evidence of a false positive. Appearance descriptors
are local tracking features, not a biometric identity or a promise of recognition.
"""
import math
import cv2
import numpy as np
from videomask.persons import iou

HUMAN={'personne','tête','visage'}
_reid=None
_feature_scope=None
_deep_features={}

def configure(media):
 global _feature_scope
 scope=str(media['path'])
 if scope!=_feature_scope:_deep_features.clear();_feature_scope=scope

def learned_feature(frame,box,index=None):
 """Use the bundled MSMT17 ReID encoder, with the official RGB / 255 preprocessing."""
 global _reid
 cache=(index,tuple(round(v) for v in box))
 if index is not None and cache in _deep_features:return _deep_features[cache]
 if _reid is None:
  from pathlib import Path
  import onnxruntime as ort
  path=Path(__file__).resolve().parent/'models/advanced/yolo26n-reid.onnx'
  if not path.exists():return None
  options=ort.SessionOptions();options.intra_op_num_threads=4;options.log_severity_level=3
  _reid=ort.InferenceSession(str(path),sess_options=options,providers=(['DmlExecutionProvider'] if 'DmlExecutionProvider' in ort.get_available_providers() else [])+['CPUExecutionProvider'])
 x,y,w,h=map(round,box);crop=frame[max(0,y):min(frame.shape[0],y+h),max(0,x):min(frame.shape[1],x+w)]
 if not crop.size:return None
 tensor=np.ascontiguousarray(cv2.resize(crop,(224,224))[:,:,::-1].transpose(2,0,1)[None]/255,dtype=np.float32)
 vector=_reid.run(None,{_reid.get_inputs()[0].name:tensor})[0].reshape(-1);vector=vector/max(float(np.linalg.norm(vector)),1e-6)
 if index is not None:
  if len(_deep_features)>=8000:_deep_features.pop(next(iter(_deep_features)))
  _deep_features[cache]=vector
 return vector

def category(obj):return (obj.get('category') or obj.get('class') or '').casefold()
def compatible(a,b):return a==b or {a,b}<={'tête','visage'}
def key(ident):return ident[3:] if ident.startswith('m-da') else ident

def descriptor(frame,box):
 x,y,w,h=map(round,box);h0,w0=frame.shape[:2];patch=frame[max(0,y):min(h0,y+h),max(0,x):min(w0,x+w)]
 if patch.size==0:return None
 tile=cv2.resize(patch,(24,40));hsv=cv2.cvtColor(tile,cv2.COLOR_BGR2HSV)
 hist=cv2.calcHist([hsv],[0,1],None,[16,8],[0,180,0,256]).flatten();hist/=max(float(np.linalg.norm(hist)),1e-6)
 gray=cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY).astype(np.float32).flatten();gray-=gray.mean();gray/=max(float(np.linalg.norm(gray)),1e-6)
 return hist,gray

def similarity(a,b):
 if a is None or b is None:return 0.
 color=float(np.dot(a[0],b[0]));texture=max(0.,float(np.dot(a[1],b[1])))
 return .35*color+.65*texture

class IdentityMemory:
 def __init__(self,frame,box,index=None,person=False):
  self.gallery=[descriptor(frame,box)];self.last=tuple(box);self.velocity=np.zeros(2);self.last_index=None
  self.person=person;feature=learned_feature(frame,box,index) if person else None;self.deep_gallery=[feature] if feature is not None else []
 def score(self,frame,box,index,fps):
  feature=descriptor(frame,box);appearance=max(similarity(d,feature) for d in self.gallery)
  x,y,w,h=box;lx,ly,lw,lh=self.last
  elapsed=abs(index-(self.last_index if self.last_index is not None else index))/max(fps,1)
  dt=max(-fps*2,min(fps*2,index-(self.last_index if self.last_index is not None else index)));predicted=np.array([lx+lw/2,ly+lh/2])+self.velocity*dt
  distance=math.hypot(x+w/2-predicted[0],y+h/2-predicted[1])/max(20,math.hypot(lw,lh)*(1+elapsed*2))
  scale=min(w*h/max(lw*lh,1),lw*lh/max(w*h,1))
  return appearance,.72*appearance+.18*math.exp(-distance)+.1*scale
 def remember(self,frame,box,index):
  if self.last_index is not None and index!=self.last_index:self.velocity=(np.array(box[:2])-np.array(self.last[:2]))/(index-self.last_index)
  self.last=tuple(box);self.last_index=index
  if index%10==0 and len(self.gallery)<10:
   self.gallery.append(descriptor(frame,box))
   feature=learned_feature(frame,box,index) if self.person else None
   if feature is not None:self.deep_gallery.append(feature)
 def choose(self,frame,objects,index,fps,lost=False):
  scored=[]
  for obj in objects:
   appearance,total=self.score(frame,obj['box'],index,fps);scored.append((total,appearance,obj))
  scored.sort(key=lambda z:z[0],reverse=True)
  if not scored:return None,False,0.
  total,appearance,best=scored[0];margin=total-scored[1][0] if len(scored)>1 else 1.
  accepted=appearance>=(.88 if lost else .55) and total>=(.83 if lost else .62) and margin>=(.12 if lost else .08)
  if not accepted and self.deep_gallery:
   learned=[]
   for score,classic,obj in scored:
    feature=learned_feature(frame,obj['box'],index)
    if feature is not None:
     cosine=max(float(feature@v) for v in self.deep_gallery);motion=(score-.72*classic)/.28;learned.append((cosine,motion,obj))
   learned.sort(key=lambda z:z[0],reverse=True)
   if learned:
    cosine,motion,candidate=learned[0];gap=cosine-learned[1][0] if len(learned)>1 else 1.
    if cosine>=.6 and gap>=.15 and motion>=.6:return candidate,False,min(.99,.89+(cosine-.6)*.25)
  return best if accepted else None,margin<(.12 if lost else .08) and total>.6,appearance

class HeadBodyBridge:
 """Body appearance can confirm a rotated head's identity, never its visibility."""
 def __init__(self,frame,head,bodies,index):
  self.memory=None;self.relative=None
  x,y,w,h=head;cx=x+w/2;cy=y+h/2
  compatible=[b for b in bodies if b['box'][0]<=cx<=b['box'][0]+b['box'][2] and b['box'][1]<=cy<=b['box'][1]+b['box'][3]*.6]
  if compatible:
   body=min(compatible,key=lambda o:o['box'][2]*o['box'][3]);bx,by,bw,bh=body['box']
   self.relative=((x-bx)/bw,(y-by)/bh,w/bw,h/bh);self.memory=IdentityMemory(frame,body['box'],index=index,person=True);self.memory.last_index=index
 def choose(self,frame,heads,bodies,index,fps,lost):
  if not self.memory or not heads:return None,0.
  body,ambiguous,appearance=self.memory.choose(frame,bodies,index,fps,lost)
  if body is None or ambiguous or appearance<.75:return None,0.
  x,y,w,h=body['box'];rx,ry,rw,rh=self.relative;expected=(x+rx*w,y+ry*h,rw*w,rh*h)
  ranked=sorted([(iou(tuple(head['box']),expected),n,head) for n,head in enumerate(heads)],reverse=True)
  if not ranked or ranked[0][0]<.25 or len(ranked)>1 and ranked[0][0]-ranked[1][0]<.15:return None,0.
  self.memory.remember(frame,body['box'],index)
  return ranked[0][2],appearance

class IndependentVerifier:
 """Lazy, independent body/face verification on a native-resolution contextual crop."""
 def __init__(self,cfg):self.cfg=cfg;self.detector=None;self.cache={}
 def check(self,media,index,box,cat,job):
  if cat not in HUMAN:return None
  if cat in {'tête','visage'}:
   x,y,w,h=box;cx=x+w/2;cy=y+h/2
   for body in getattr(self,'body_evidence',{}).get(index,[]):
    bx,by,bw,bh=body['box']
    if body.get('confidence',0)>=.85 and bx<=cx<=bx+bw and by<=cy<=by+bh*.35 and .06<=w/max(bw,1)<=.9:return True
  cache=(index,tuple(round(v/12) for v in box),cat)
  if cache in self.cache:return self.cache[cache]
  if job.get('cancel'):raise InterruptedError('Vérification annulée.')
  if self.detector is None:
   from engine import Detector
   cfg=self.cfg.model_copy(update={'scope':'tous','heads':False,'faces':True,'classes':['personne'],'extended':[],'ocr':False,'brands':False,'plates':False,'grounding_prompts':[],'profile':'moyen'})
   self.detector=Detector(cfg)
  from inspection import source_at
  frame=source_at(media,index);x,y,w,h=box;pad=max(w,h)*(.9 if cat!='personne' else .25)
  xa=max(0,round(x-pad));ya=max(0,round(y-pad*.45));xb=min(frame.shape[1],round(x+w+pad));yb=min(frame.shape[0],round(y+h+pad*1.7))
  crop=frame[ya:yb,xa:xb];target=(x-xa,y-ya,w,h);found=False
  if crop.size:
   for obj in self.detector.detect(crop):
    bx,by,bw,bh=obj['box'];cx=x-xa+w/2;cy=y-ya+h/2
    face=obj['class']=='visage' and (iou(tuple(obj['box']),target)>.12 or (cat=='personne' and x-xa<=bx+bw/2<=x-xa+w and y-ya<=by+bh/2<=y-ya+h*.6))
    body=obj['class']=='personne' and obj.get('confidence',0)>=.6 and (iou(tuple(obj['box']),target)>.35 if cat=='personne' else bx<=cx<=bx+bw and by<=cy<=by+bh*.5)
    if face or body:found=True;break
  self.cache[cache]=True if found else None
  return self.cache[cache]

 def deep_check(self,media,index,box,cat,job):
  if self.detector is None or not hasattr(self.detector,'deep'):return None
  from inspection import source_at
  frame=source_at(media,index);x,y,w,h=box;pad=max(w,h)*(.9 if cat!='personne' else .25);xa=max(0,round(x-pad));ya=max(0,round(y-pad*.45));crop=frame[ya:min(frame.shape[0],round(y+h+pad*1.7)),xa:min(frame.shape[1],round(x+w+pad))]
  if not crop.size:return None
  target=(x-xa,y-ya,w,h)
  for obj in self.detector.deep(crop,[(0,0,crop.shape[1],crop.shape[0])],thorough=True):
   bx,by,bw,bh=obj['box'];face=obj['class']=='visage' and (iou(tuple(obj['box']),target)>.12 or cat=='personne' and target[0]<=bx+bw/2<=target[0]+w and target[1]<=by+bh/2<=target[1]+h*.6);body=obj['class']=='personne' and obj.get('confidence',0)>=.6 and (iou(tuple(obj['box']),target)>.35 if cat=='personne' else bx<=target[0]+w/2<=bx+bw and by<=target[1]+h/2<=by+bh*.5)
   if face or body:return True
  return None


def reconcile_automatic_tracks(media,frames,cuts,job,bodies=None):
 """Join short, non-overlapping trajectories only with reciprocal unique evidence."""
 from inspection import source_at
 tracks={}
 for index,row in enumerate(frames):
  for obj in row:
   if obj['id'].startswith('a') and obj.get('detected',True):tracks.setdefault(obj['id'],[]).append((index,obj))
 features={};deep={};candidates=[]
 from bisect import bisect_right
 starts=sorted((rows[0][0],ident) for ident,rows in tracks.items());start_indices=[i for i,_ in starts]
 def learned(index,obj):
  cache=(index,obj['id'])
  if cache not in deep:deep[cache]=learned_feature(source_at(media,index),obj['box'],index)
  return deep[cache]
 def feature(index,obj):
  cache=(index,obj['id'])
  if cache not in features:features[cache]=descriptor(source_at(media,index),obj['box'])
  return features[cache]
 def body_feature(index,obj):
  if category(obj)=='personne':return learned(index,obj)
  x,y,w,h=obj['box'];cx=x+w/2;cy=y+h/2
  candidates=[o for o in (bodies or {}).get(index,[]) if o['box'][0]<=cx<=o['box'][0]+o['box'][2] and o['box'][1]<=cy<=o['box'][1]+o['box'][3]*.65]
  if len(candidates)!=1:return None
  body=candidates[0];bx,by,bw,bh=body['box']
  heads=[o for o in frames[index] if category(o) in {'tête','visage'} and bx<=o['box'][0]+o['box'][2]/2<=bx+bw and by<=o['box'][1]+o['box'][3]/2<=by+bh*.65]
  if len(heads)!=1:return None
  return learned(index,dict(body,id=obj['id']+'-body'))
 from performance import phase,tick
 phase('Tracking',len(tracks),'trajectoires')
 for ordinal,(before,rows) in enumerate(tracks.items(),1):
  tick(ordinal)
  end,last=rows[-1]
  horizon=min((c for c in cuts if c>end),default=media['frames']) if category(last) in HUMAN else end+media['fps']*5
  for _,after in starts[bisect_right(start_indices,end):bisect_right(start_indices,horizon)]:
   later=tracks[after];start,head=later[0]
   if before==after or start<=end or not compatible(category(last),category(head)) or any(end<c<=start for c in cuts):continue
   if job.get('cancel'):raise InterruptedError('Vérification des identités annulée.')
   lx,ly,lw,lh=last['box'];x,y,w,h=head['box'];velocity=np.zeros(2)
   if len(rows)>1:
    previous,prior=rows[-2];velocity=(np.array(last['box'][:2])-np.array(prior['box'][:2]))/max(1,end-previous)
   expected=np.array([lx+lw/2,ly+lh/2])+velocity*min(start-end,media['fps']*2)
   distance=float(np.linalg.norm(np.array([x+w/2,y+h/2])-expected))/max(20,math.hypot(lw,lh)*(1+(start-end)/media['fps']*2))
   scale=min(w*h/max(1,lw*lh),lw*lh/max(1,w*h))
   if scale<(.12 if category(last) in HUMAN else .3) or distance>1.5:continue
   # Compare several native crops and a motion prediction. Similar candidates
   # on either side of the gap prevent an automatic association.
   appearance=max(similarity(feature(i,o),feature(start,head)) for i,o in rows[-3:])
   deep_similarity=None
   if category(last) in HUMAN:
    target=body_feature(start,head)
    previous=[body_feature(i,o) for i,o in rows[-3:]]
    if target is not None:deep_similarity=max((float(target@v) for v in previous if v is not None),default=0.)
   minimum_deep=.7 if category(last) in {'tête','visage'} or scale<.3 else .6
   if start-end>media['fps']*5:
    # Long re-entries need learned appearance and a much stronger match;
    # similar clothes or a nearby position alone are not an identity proof.
    if deep_similarity is None or deep_similarity<.86:continue
    minimum_deep=max(minimum_deep,.86)
   if appearance<.88 and (deep_similarity is None or deep_similarity<minimum_deep):continue
   score=.72*appearance+.18*math.exp(-distance)+.1*scale+.08*math.exp(-(start-end)/max(1,media['fps']*.2))
   motion=(score-.72*appearance)/.28
   if deep_similarity is not None and deep_similarity>=minimum_deep and motion>=.6:score=max(score,.9+(deep_similarity-minimum_deep)*.25);appearance=max(appearance,score)
   if score>=.83:candidates.append((score,appearance,before,after,start))
 accepted=[]
 for candidate in sorted(candidates,reverse=True):
  score,appearance,before,after,start=candidate
  competitors=[s for s,_,a,b,_ in candidates if (a==before or b==after) and (a,b)!=(before,after)]
  if competitors and score-max(competitors)<.04:continue
  if any(a==before or b==after for _,_,a,b,_ in accepted):continue
  accepted.append(candidate)
 parent={}
 for score,appearance,before,after,start in sorted(accepted,key=lambda c:c[4]):
  while before in parent:before=parent[before]
  parent[after]=before
  for i,obj in tracks[after]:
   obj.setdefault('reference_ids',[])
   obj['reference_ids']=list(set([*obj['reference_ids'],after]))
   obj['id']=before;obj['color']=tracks[before][0][1]['color']
   if i==start:obj.update(reidentified=True,appearance_confidence=appearance)
 # Skipped detector frames may contain CSRT copies carrying the former ID.
 for row in frames:
  for obj in row:
   ident=obj['id']
   while ident in parent:ident=parent[ident]
   if ident!=obj['id']:obj['reference_ids']=list(set([*obj.get('reference_ids',[]),obj['id']]));obj['id']=ident;obj['color']=tracks[ident][0][1]['color']
 return len(accepted)

def spans(values,fps):
 result=[]
 for index,state in values:
  if result and result[-1]['state']==state and result[-1]['last']+1==index:result[-1]['last']=index;result[-1]['end']=(index+1)/fps
  else:result.append({'state':state,'start':index/fps,'end':(index+1)/fps,'first':index,'last':index})
 return result

def evaluate_subject(media,seed,observations,anchors,absent,cuts,first,last,verifier,job):
 """Retrospective confirmation preserves strong one-frame detections and candidates."""
 fps=media['fps'];cat=(seed.category or seed.label).casefold();manual=not seed.id.startswith(('m-da','a'))
 evidence=[(i,o) for i,o in sorted(observations.items()) if o.get('detected') and o.get('visible')]
 # Repeated coherent detections can corroborate a weak initial observation, but
 # frame count alone never rejects a short track.
 confirmed=manual or any(o.get('existence_verified') or o.get('detection_confidence',0)>=.85 and not o.get('confidence_estimated') and not o.get('verification_uncertain') for _,o in evidence)
 if not confirmed and evidence:
  shots=[(-1,*cuts,media['frames'])] # use only evidence from the same shot
  for a,b in zip(shots[0],shots[0][1:]):
   rows=[(i,o) for i,o in evidence if a<=i<b and not o.get('verification_uncertain') and o.get('detection_confidence',0)>=(.55 if cat=='personne' else .5)]
   if len(rows)>=max(3,round(fps*.35)) and any(not o.get('confidence_estimated') for _,o in rows) and max(i for i,_ in rows)-min(i for i,_ in rows)>=min(2,max(1,round(fps*.08))):confirmed=True;break
 promising=len(evidence)>=3 and max([o.get('detection_confidence',0) for _,o in evidence] or [0])>=.65
 catalogue=getattr(getattr(verifier,'cfg',None),'catalog_only',False)
 if not confirmed and evidence and (not catalogue or promising):
  job['phase']='Vérification indépendante · '+seed.label
  for attempt,(index,obj) in enumerate(sorted(evidence,key=lambda z:z[1].get('detection_confidence',0),reverse=True)[:3]):
   found=verifier.check(media,index,obj['box'],cat,job)
   if not found and attempt==0 and getattr(getattr(verifier,'cfg',None),'profile',None)=='extreme' and hasattr(verifier,'deep_check'):found=verifier.deep_check(media,index,obj['box'],cat,job)
   if found:confirmed=True;obj['independent_confirmation']=True;break
 # A selected semantic seed is still a candidate, unless its local appearance
 # is supported across time or an independent detector confirms it.
 if not confirmed and not evidence and observations and cat not in HUMAN:
  confirmed=manual
 if confirmed and cat in HUMAN:
  count=__import__('privacy_continuity').protect(media,observations,cuts,absent,getattr(verifier,'body_evidence',{}))
  job['privacy_coverage_frames']=job.get('privacy_coverage_frames',0)+count
 states={};rows=[];issues=[];visible=sorted(i for i,o in observations.items() if o.get('visible') and o.get('tracking_confidence',0)>=.6)
 from temporal_presence import edge_motion
 exited=bool(visible and edge_motion(media,observations,visible[-1],True));occluded=set();outside=set()
 if confirmed:
  for a,b in zip(visible,visible[1:]):
   if b-a<=1 or any(a<c<=b for c in cuts):continue
   # Only a unique, high-appearance re-identification can close an occlusion.
   if observations[b].get('reidentified') and observations[b].get('appearance_confidence',0)>=.88:
    if edge_motion(media,observations,a,True) and edge_motion(media,observations,b,False):outside.update(range(a+1,b))
    elif b-a<=fps*5:occluded.update(range(a+1,b))
 for index in range(first,last):
  obj=observations.get(index)
  if index in absent:state='out_of_frame' if exited and index>visible[-1] else 'absent'
  elif index in outside:state='out_of_frame'
  elif index in occluded:state='occluded'
  elif obj and obj.get('visible'):
   state='visible' if confirmed and obj.get('tracking_confidence',0)>=.6 and not obj.get('identity_ambiguous') else 'doubt' if confirmed else 'candidate'
  else:state='doubt'
  states[index]=state
  if obj:
   row=dict(obj,id=seed.id,reference_ids=list(getattr(seed,'reference_ids',[])),**{'class':seed.label,'category':seed.category,'color':seed.color})
   row.update(timestamp=index/fps,trajectory_id=seed.id,visibility_confidence=(1. if obj.get('detected') else obj.get('tracking_confidence',0.)) if obj.get('visible') else 0.,existence='confirmed' if confirmed else 'candidate',presence=state,mask_allowed=state=='visible',tracking_status='ok' if state=='visible' else 'review')
   rows.append((index,row))
 for span in spans(sorted(states.items()),fps):
  if span['state'] not in {'doubt','candidate'}:continue
  reason='Détection à confirmer' if span['state']=='candidate' else 'Identité ou visibilité incertaine'
  issue={'id':f'{seed.id}:{span["first"]}:{span["last"]}:{reason}','subject_id':seed.id,'label':seed.label,'start':span['start'],'end':span['end'],'reason':reason,'kind':'detection' if span['state']=='candidate' else 'tracking','reference_time':seed.time,'reference_box':seed.box,'reference_ids':list(getattr(seed,'reference_ids',[])),'color':seed.color,'category':seed.category}
  if span['first'] in observations:
   o=observations[span['first']];issue.update(detection_confidence=o.get('detection_confidence',0),tracking_confidence=o.get('tracking_confidence',0),box=[v/n*100 for v,n in zip(o['box'],[media['width'],media['height'],media['width'],media['height']])])
  issues.append(issue)
 trajectory={'id':seed.id,'label':seed.label,'category':seed.category,'color':seed.color,'existence':'confirmed' if confirmed else 'candidate','reference_time':seed.time,'reference_box':seed.box,'intervals':spans(sorted(states.items()),fps)}
 return rows,issues,trajectory

def decisions_for(cfg,ident,time):
 return [d for d in cfg.temporal_decisions if key(d.subject_id)==key(ident) and d.start-1e-7<=time<d.end-1e-7]

def resolve_object(original,cfg,time):
 original=__import__('linear_tracking').selected(original,cfg)
 if original is None:return None
 obj=dict(original);seen={key(obj['id'])}
 # Explicit links/splits are reversible rendering decisions. No inferred merge.
 for _ in range(8):
  related={key(obj['id']),*[key(r) for r in obj.get('reference_ids',[])]}
  decision=next((d for d in reversed(cfg.temporal_decisions) if key(d.subject_id) in related and d.start-1e-7<=time<d.end-1e-7 and d.action in {'link','split'} and d.target_id),None)
  if not decision or key(decision.target_id) in seen:break
  obj.setdefault('source_subject_id',obj['id']);obj['id']=decision.target_id;seen.add(key(obj['id']))
  if obj.get('visible') and obj.get('tracking_confidence',0)>=.6:obj.update(identity_ambiguous=False,mask_allowed=True,presence='visible')
 related={key(original['id']),key(obj['id']),*[key(r) for r in obj.get('reference_ids',[])]}
 for d in [d for d in cfg.temporal_decisions if key(d.subject_id) in related and d.start-1e-7<=time<d.end-1e-7]:
  if d.action in {'reject','suppress','absent','occluded'}:return None
  if d.action=='contour' and d.polygon and obj.get('visible'):
   x,y,w,h=obj['box'];obj.update(polygons=[[[x+px*w/100,y+py*h/100] for px,py in d.polygon]],manual_contour=True,mask_allowed=True,existence='confirmed',presence='visible',identity_ambiguous=False)
  if d.action=='confirm' and obj.get('visible') and not obj.get('identity_ambiguous') and obj.get('tracking_confidence',0)>=.6:obj.update(existence='confirmed',presence='visible',mask_allowed=True)
 # Selected, visible doubtful regions stay covered in preview and export.
 # Advisory review remains unresolved; rendering does not confirm identity.
 selected={key(v) for v in cfg.selected_ids}|{key(s.id) for s in cfg.custom if not s.hidden and not s.tracking_only}
 if not obj.get('mask_allowed',True) and obj.get('visible') and obj.get('presence') in {'candidate','doubt'} and related.intersection(selected):obj.update(mask_allowed=True,privacy_coverage=True,protected_uncertain=True)
 if original.get('presence')=='absent' or not obj.get('mask_allowed',True):return None
 return obj

def identity_at(cfg,ident,time):
 seen={key(ident)}
 for _ in range(8):
  d=next((d for d in reversed(decisions_for(cfg,ident,time)) if d.action in {'link','split'} and d.target_id),None)
  if not d or key(d.target_id) in seen:break
  ident=d.target_id;seen.add(key(ident))
 return ident

def resolved_review(plan,cfg,fps):
 from tracking import gap_confirmed
 result=[];selected={key(k) for k in cfg.selected_ids};seeds={key(s.id):s for s in cfg.custom}
 allowed=selected|{key(s.id) for s in cfg.custom if not s.hidden and not s.tracking_only}
 for gap in plan.get('review',[]):
  start=round(gap['start']*fps);end=round(gap['end']*fps);boundaries={start,end}
  for d in cfg.temporal_decisions:
   for at in [round(d.start*fps),round(d.end*fps)]:
    if start<at<end:boundaries.add(at)
  for value in cfg.confirmed_gaps:
   try:_,a,b,_=value.rsplit(':',3);a=int(a);b=int(b)+1
   except (ValueError,AttributeError):continue
   for at in [a,b]:
    if start<at<end:boundaries.add(at)
  edges=sorted(boundaries)
  for a,b in zip(edges,edges[1:]):
   time=(a+b)/2/fps;ident=identity_at(cfg,gap['subject_id'],time);seed=seeds.get(key(gap['subject_id']))
   if seed and seed.tracking_only and key(ident) not in selected:continue
   aliases={key(gap['subject_id']),key(ident),*[key(r) for r in gap.get('reference_ids',[])]}
   if (cfg.library_only or cfg.scope=='selection') and not aliases&allowed:continue
   g=dict(gap,subject_id=ident,start=a/fps,end=b/fps,reference_ids=sorted(aliases))
   if ident!=gap['subject_id']:
    override=cfg.subject_overrides.get(ident,{}) or cfg.subject_overrides.get(key(ident),{});target=seeds.get(key(ident));g['label']=override.get('class',target.label if target else gap['label'])
   if a!=start or b!=end or ident!=gap['subject_id']:g['id']=f'{ident}:{a}:{b-1}:{gap["reason"]}'
   if gap_confirmed(g,cfg.confirmed_gaps,fps):continue
   decisions=[d for d in cfg.temporal_decisions if key(d.subject_id) in aliases and d.start-1e-7<=time<d.end-1e-7]
   resolved=any(d.action in {'reject','suppress','absent','occluded'} or d.action in {'confirm','contour'} and gap.get('kind')=='detection' or d.action in {'link','split'} and gap.get('kind')=='identity' for d in decisions)
   if resolved:continue
   if result and result[-1]['subject_id']==g['subject_id'] and result[-1]['reason']==g['reason'] and abs(result[-1]['end']-g['start'])<.1/fps:
    previous=result[-1];previous['end']=g['end'];previous['id']=f'{ident}:{round(previous["start"]*fps)}:{b-1}:{gap["reason"]}'
   else:result.append(g)
 return result

def stabilize(frames,cuts):
 """Short, bounded box stabilizer; never bridges an invisible frame or a cut."""
 previous={}
 for i,row in enumerate(frames):
  if i in cuts:previous={}
  next_previous={}
  for obj in row:
   if not obj.get('visible',True) or obj.get('tracking_confidence',1)<.6 or obj.get('identity_ambiguous'):continue
   raw=np.asarray(obj['box'],float);last=previous.get(obj['id'])
   if last is not None and iou(tuple(raw),tuple(last))>.65:
    filtered=.7*raw+.3*last
    # Enclose the raw detection as well, so smoothing cannot uncover its edge.
    x=min(raw[0],filtered[0]);y=min(raw[1],filtered[1]);right=max(raw[0]+raw[2],filtered[0]+filtered[2]);bottom=max(raw[1]+raw[3],filtered[1]+filtered[3])
    obj['box']=[round(x),round(y),round(right-x),round(bottom-y)]
   next_previous[obj['id']]=raw
  previous=next_previous
