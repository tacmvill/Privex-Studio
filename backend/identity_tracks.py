"""Online geometric/appearance association over detector tracklets, bounded by active tracks."""
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from videomask.persons import iou

def appearance(frame,box):
 if frame is None:return None
 x,y,w,h=map(round,box);crop=frame[max(0,y):min(frame.shape[0],y+h),max(0,x):min(frame.shape[1],x+w)]
 if not crop.size:return None
 hsv=cv2.cvtColor(cv2.resize(crop,(24,32)),cv2.COLOR_BGR2HSV)
 hist=cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256]).reshape(-1);norm=np.linalg.norm(hist)
 return hist/max(1e-6,norm)

def head_groups(heads):
 # Several detectors can report the same head inside one body. That should
 # not invalidate otherwise unique clothing/body context.
 groups=[]
 for obj in heads:
  if not any(iou(tuple(obj['box']),tuple(other['box']))>=.6 for other in groups):groups.append(obj)
 return len(groups)

class IdentityTracker:
 def __init__(self,inner,fps,namespace='',motion_fps=None,metrics=None):
  self.inner=inner;self.fps=fps;self.namespace=namespace;self.index=0;self.serial=0;self.active={};self.pending=[];self.aliases={};self.colors={};self.context_gallery={};self.source_index=None
  self.motion=__import__('sparse_motion').SparseMotion(motion_fps or fps);self.stats=metrics if metrics is not None else {};self.stats.setdefault('measured_motion_frames',0);self.stats.setdefault('body_assisted_frames',0);self.stats.setdefault('low_score_continuations',0)
 def update(self,objects,frame=None,auxiliary=None):
  from engine import PALETTE
  motion=self.motion.infer(frame) if frame is not None else [];predictions={o['id']:o for o in motion}
  from sparse_motion import feature as visual_feature
  # A detector may call the same moving vehicle bus, truck or car as its
  # viewing angle changes. Alternative classes may support an existing selected
  # trajectory only; they cannot create another subject or broaden discovery.
  resolved=[]
  for obj in objects:
   if not obj.get('vehicle_continuation'):
    resolved.append(obj);continue
   ranked=[]
   for ident,state in self.active.items():
    if state['class_id'] not in {2,5,7} or self.index-state['at']>self.fps*2:continue
    predicted=predictions.get(ident,{}).get('box',state['box']);overlap=iou(tuple(obj['box']),tuple(predicted))
    if overlap<.25:continue
    current=visual_feature(frame,obj['box']) if frame is not None else None;reference=state.get('visual')
    similarity=.6*float(current[0]@reference[0])+.4*max(0.,float(current[1]@reference[1])) if current is not None and reference is not None else 0.
    partial_box=None
    # Match the visible subregion to the last full vehicle view. Comparing a
    # half-occluded cab to a whole truck otherwise fails at an occluder.
    view=state.get('vehicle_view')
    if view and self.index-view['at']<=max(2,round(self.fps*.24)) and obj['confidence']>=.1:
     # Detector crop margins can move while the vehicle itself does not. Test
     # both measured translation and the last view, with appearance evidence.
     for displacement in [view['velocity']*(self.index-view['at']),np.zeros(2)]:
      full=np.asarray(view['box'],float).copy();full[:2]+=displacement;x,y,w,h=obj['box'];ox,oy,ow,oh=full
      ratio=w*h/max(1,ow*oh)
      if not (.15<=ratio<=.8 and iou(tuple(full),tuple(obj['box']))>=.25):continue
      reference_part=visual_feature(view['image'],[(x-ox)/ow*96,(y-oy)/oh*96,w/ow*96,h/oh*96])
      if reference_part is None or current is None:continue
      color=float(current[0]@reference_part[0]);texture=max(0.,float(current[1]@reference_part[1]))
      if color>=.85 and texture>=.25 and .6*color+.4*texture>=.67:
       left,top=min(x,ox),min(y,oy);right,bottom=max(x+w,ox+ow),max(y+h,oy+oh);partial_box=[round(left),round(top),round(right-left),round(bottom-top)];break
    if similarity<.6 and ident not in predictions and partial_box is None:continue
    ranked.append((overlap,ident,partial_box))
   ranked.sort(reverse=True)
   if not ranked or len(ranked)>1 and ranked[0][0]-ranked[1][0]<.15:continue
   _,ident,partial_box=ranked[0];state=self.active[ident];source=state['obj']
   if any(not o.get('vehicle_continuation') and o['class_id']==state['class_id'] and iou(tuple(o['box']),tuple(obj['box']))>.45 for o in objects):continue
   if partial_box is not None:obj=dict(obj,box=partial_box,vehicle_partial=True,privacy_coverage=True)
   resolved.append(dict(obj,vehicle_target=ident,model_category=obj.get('category',obj['class']),class_id=state['class_id'],category=source.get('category',source['class']),**{'class':source['class']}))
  objects=resolved
  def continuation(obj):
   if not obj.get('continuation_only') or obj.get('vehicle_target') in self.active:return True
   visual=visual_feature(frame,obj['box']) if frame is not None else None
   for ident,state in self.active.items():
    if obj['class_id']!=state['class_id'] or iou(tuple(obj['box']),tuple(predictions.get(ident,{}).get('box',state['box'])))<=.1:continue
    if ident in predictions:return True
    reference=state.get('visual')
    if visual is not None and reference is not None and .6*float(visual[0]@reference[0])+.4*max(0.,float(visual[1]@reference[1]))>=.75:return True
   return False
  objects=[o for o in objects if continuation(o)]
  rows=self.inner.update(objects,frame);self.index+=1;age=max(2,round(self.fps*30));self.active={k:v for k,v in self.active.items() if self.index-v['at']<=age}
  # Predicted boxes without a matching detection do not create another identity.
  rows=[o for o in rows if o.get('detected',True)];ids=list(self.active);cost=np.full((len(ids),len(rows)),1e3);features={}
  body_cache={}
  def body_match(obj,state):
   if frame is None or state.get('body_feature') is None or state.get('body') is None:return 0.
   x,y,w,h=obj['box'];candidates=[b for b in auxiliary or [] if b['box'][0]<=x+w/2<=b['box'][0]+b['box'][2] and b['box'][1]<=y+h/2<=b['box'][1]+b['box'][3]*.6]
   if len(candidates)!=1:return 0.
   body=candidates[0];bx,by,bw,bh=body['box'];heads=[o for o in objects if o['class_id']==obj['class_id'] and bx<=o['box'][0]+o['box'][2]/2<=bx+bw and by<=o['box'][1]+o['box'][3]/2<=by+bh*.6]
   if head_groups(heads)!=1 or iou(tuple(body['box']),tuple(state['body']))<.25:return 0.
   key=tuple(body['box'])
   if key not in body_cache:body_cache[key]=visual_feature(frame,body['box'])
   current=body_cache[key];previous=state['body_feature']
   return .6*float(current[0]@previous[0])+.4*max(0.,float(current[1]@previous[1])) if current is not None else 0.
  assigned={};claimed=set();uncertain=set()
  for b,obj in enumerate(rows):
   known=obj.get('vehicle_target') or self.aliases.get(obj['id'])
   if known in predictions and iou(tuple(obj['box']),tuple(predictions[known]['box']))<.1 and any(k!=known and iou(tuple(obj['box']),tuple(p['box']))>.6 for k,p in predictions.items()):known=None
   if known in self.active and frame is not None:
    features[b]=appearance(frame,obj['box']);reference=self.active[known]['appearance']
    if features[b] is not None and reference is not None and float(features[b]@reference)<.55 and body_match(obj,self.active[known])<.9:
     state=self.active[known];predicted=predictions.get(known,{}).get('box',state['box'])
     if self.index-state['at']>2 or iou(tuple(obj['box']),tuple(predicted))<.35:known=None
   if known in self.active and known not in claimed:assigned[b]=known;claimed.add(known)
  for a,ident in enumerate(ids):
   if ident in claimed:continue
   state=self.active[ident];gap=self.index-state['at'];old=np.asarray(state['box'],float);expected=old.copy();expected[:2]+=state['velocity']*min(self.index-state.get('updated',state['at']),self.fps*.3)
   if ident in predictions:expected=np.asarray(predictions[ident]['box'],float)
   for b,obj in enumerate(rows):
    if b in assigned:continue
    if obj['class_id']!=state['class_id']:continue
    box=np.asarray(obj['box'],float);overlap=iou(tuple(expected),tuple(box));ratio=min(box[2]*box[3]/max(1,old[2]*old[3]),old[2]*old[3]/max(1,box[2]*box[3]));distance=np.linalg.norm(box[:2]+box[2:]/2-expected[:2]-expected[2:]/2)/max(10,np.linalg.norm(old[2:]))
    standard=ratio>=.35 and distance<=.85 and (overlap>=.08 or distance<=.35)
    recovered=False
    if not standard and frame is not None and 1<gap<=self.fps*1.5 and ratio>=.5 and distance<=1.8:
     candidate=visual_feature(frame,obj['box']);gallery=state.get('visual_gallery',[state.get('visual')])
     recovered=bool(candidate is not None and any(reference is not None and .35*float(candidate[0]@reference[0])+.65*max(0.,float(candidate[1]@reference[1]))>=.92 for reference in gallery))
    if not standard and not recovered:continue
    same=obj['id']==state['upstream'];score=.65*overlap+.2*np.exp(-distance*3)+.15*ratio
    if frame is not None:
     if b not in features:features[b]=appearance(frame,obj['box'])
     similarity=float(features[b]@state['appearance']) if features[b] is not None and state['appearance'] is not None else .5
     body_similarity=body_match(obj,state)
     if body_similarity>=.9:similarity=max(similarity,body_similarity)
     if similarity<.5 or not same and gap>1 and similarity<.65:continue
     score=.7*score+.3*similarity
    if recovered:score=max(score,.82)
    if same:score+=.12
    score-=min(.25,(gap-1)*.055)
    if score>=.48:cost[a,b]=1-score
  if cost.size:
   aa,bb=linear_sum_assignment(cost)
   for a,b in zip(aa,bb):
    if cost[a,b]>=1:continue
    alternatives=[cost[c,b] for c in range(len(ids)) if c!=a and cost[c,b]<1]+[cost[a,j] for j in range(len(rows)) if j!=b and cost[a,j]<1]
    # A crossing with equally plausible identities remains explicit, never a silent merge.
    continuity=rows[b]['id']==self.active[ids[a]]['upstream'] and self.index-self.active[ids[a]]['at']==1
    if not continuity and alternatives and min(alternatives)-cost[a,b]<.04:uncertain.add(b)
    assigned[b]=ids[a]
  result=[]
  for b,obj in enumerate(rows):
   ident=assigned.get(b)
   if ident is None and obj.get('continuation_only'):continue
   if ident is None:self.serial+=1;ident=f'a{self.namespace}u{self.serial}'
   previous=self.active.get(ident);box=np.asarray(obj['box'],float);velocity=(box[:2]-np.asarray(previous['box'][:2]))/max(1,self.index-previous.get('updated',previous['at'])) if previous else np.zeros(2)
   color=self.colors.get(ident)
   if color is None:color=PALETTE[len(self.colors)%len(PALETTE)];self.colors[ident]=color
   feature=features.get(b)
   if feature is None:feature=appearance(frame,obj['box'])
   confidence=float(feature@previous['appearance']) if previous and feature is not None and previous['appearance'] is not None else 1.
   gallery=[*(previous.get('visual_gallery',[]) if previous else [])]
   returned=bool(previous and self.index-previous['at']>1)
   history=[*(previous.get('history',[]) if previous else []),obj['box']][-4:]
   visual=previous.get('visual') if previous and obj.get('continuation_only') and not (obj.get('vehicle_target') and obj.get('confidence',0)>=.3) else visual_feature(frame,obj['box']) if frame is not None else None
   self.active[ident]={'at':self.index,'updated':self.index,'obj':dict(obj,id=ident,color=color),'visual':visual,'history':history,'status':'visible','body':previous.get('body') if previous else None,'body_feature':previous.get('body_feature') if previous else None,'box':obj['box'],'velocity':velocity,'class_id':obj['class_id'],'upstream':obj['id'],'appearance':feature,'color':color}
   if visual is not None and (len(gallery)<3 or self.index%10==0):gallery.append(visual)
   self.active[ident]['visual_gallery']=gallery[-6:]
   view=previous.get('vehicle_view') if previous else None
   if frame is not None and obj['class_id'] in {2,5,7} and obj.get('confidence',0)>=.3 and not obj.get('vehicle_partial') and b not in uncertain:
    x,y,w,h=map(round,obj['box']);crop=frame[max(0,y):min(frame.shape[0],y+h),max(0,x):min(frame.shape[1],x+w)]
    if crop.size:view={'at':self.index,'box':list(obj['box']),'velocity':np.median(np.diff(np.asarray(history,float)[:,:2],axis=0),axis=0) if len(history)>1 else np.zeros(2),'image':cv2.resize(crop,(96,96))}
   if view is not None:self.active[ident]['vehicle_view']=view

   self.aliases[obj['id']]=ident
   for old in list(self.active):
    if old!=ident and old not in assigned.values() and self.active[old]['upstream']==obj['id']:del self.active[old]
   result.append(dict(obj,id=ident,color=color,identity_ambiguous=b in uncertain or obj.get('identity_ambiguous',False),reidentified=returned,appearance_confidence=confidence))
   if obj.get('continuation_only'):self.stats['low_score_continuations']+=1
  if frame is not None:
   bodies=auxiliary or [];body_features={}
   from sparse_motion import feature
   def body_feature(b):
    key=tuple(b['box'])
    if key not in body_features:body_features[key]=feature(frame,b['box'])
    return body_features[key]
   for obj in result:
    if obj.get('category',obj['class']) not in {'tête','visage'}:continue
    x,y,w,h=obj['box'];candidates=[b for b in bodies if b['box'][0]<=x+w/2<=b['box'][0]+b['box'][2] and b['box'][1]<=y+h/2<=b['box'][1]+b['box'][3]*.6]
    if len(candidates)==1:
     body=candidates[0];bx,by,bw,bh=body['box'];heads=[o for o in result if o.get('category',o['class']) in {'tête','visage'} and bx<=o['box'][0]+o['box'][2]/2<=bx+bw and by<=o['box'][1]+o['box'][3]/2<=by+bh*.6]
     if head_groups(heads)==1:self.active[obj['id']].update(body=body['box'],body_feature=body_feature(body))
   claimed={o['id'] for o in result}
   for ident,state in self.active.items():
    if ident in claimed:continue
    inferred=predictions.get(ident)
    if inferred is not None and not any(o['class_id']==state['class_id'] and iou(tuple(o['box']),tuple(inferred['box']))>.3 for o in result):
     result.append(inferred);state.update(box=inferred['box'],updated=self.index,status='measured');self.stats['measured_motion_frames']+=1;continue
    # A unique observed body can keep protecting a turned head. It cannot
    # confirm that a face is visible, and never creates a new subject.
    ranked=[]
    if state.get('body_feature') is not None and self.index-state['at']<=self.fps*8:
     old=state['body'];ox,oy,ow,oh=old
     for body in bodies:
      bf=body_feature(body)
      if bf is None:continue
      color=float(bf[0]@state['body_feature'][0]);texture=max(0.,float(bf[1]@state['body_feature'][1]));similarity=.6*color+.4*texture
      if similarity<.88 or iou(tuple(old),tuple(body['box']))<.15:continue
      bx,by,bw,bh=body['box']
      if any(o.get('category',o['class']) in {'tête','visage'} and bx<=o['box'][0]+o['box'][2]/2<=bx+bw and by<=o['box'][1]+o['box'][3]/2<=by+bh*.6 for o in result):continue
      ranked.append((similarity,body))
    ranked.sort(key=lambda pair:pair[0],reverse=True)
    if ranked and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.1):
     similarity,body=ranked[0];bx,by,bw,bh=body['box'];source=state['obj'];hx,hy,hw,hh=source['box'];rx=(hx-ox)/max(1,ow);ry=(hy-oy)/max(1,oh);head_box=[round(bx+rx*bw),round(by+ry*bh),round(max(.05,min(1.,hw/max(1,ow)))*bw),round(max(.08,min(.45,hh/max(1,oh)))*bh)];obj=dict(source,id=ident,box=head_box,detected=False,visible=True,detection_confidence=0.,tracking_confidence=.8,appearance_confidence=similarity,body_verified=True,privacy_coverage=True)
     result.append(obj);state.update(box=obj['box'],body=body['box'],updated=self.index,status='body-supported');self.stats['body_assisted_frames']+=1
    else:
     x,y,w,h=state['box'];vx,vy=state['velocity'];height,width=frame.shape[:2]
     exited=len(state.get('history',[]))>=3 and (x<=2 and vx<-.5 or y<=2 and vy<-.5 or x+w>=width-2 and vx>.5 or y+h>=height-2 and vy>.5)
     overlap=lambda box:max(0,min(x+w,box[0]+box[2])-max(x,box[0]))*max(0,min(y+h,box[1]+box[3])-max(y,box[1]))/max(1,w*h)
     occluded=any(o['id']!=ident and overlap(o['box'])>.65 and o['box'][2]*o['box'][3]>w*h*1.2 for o in result)
     status='out_of_frame' if exited else 'occluded' if occluded else 'occluded-or-lost'
     if state.get('status')!=status:self.stats[status+'_episodes']=self.stats.get(status+'_episodes',0)+1
     state['status']=status
   from context_association import remember
   for obj in result:
    if not obj.get('detected',True) or obj.get('identity_ambiguous'):continue
    state=self.active.get(obj['id'])
    if state:remember(self.context_gallery,obj['id'],self.source_index if self.source_index is not None else self.index,obj,state.get('visual'),state.get('body_feature'))
   self.motion.commit(result,frame)
  return result

 def predict(self,frame):
  rows=self.motion.infer(frame);self.motion.commit(rows,frame);self.stats['measured_motion_frames']+=len(rows)
  for obj in rows:
   if obj['id'] in self.active:self.active[obj['id']].update(box=obj['box'],updated=self.index,status='measured')
  return rows
