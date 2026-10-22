"""Retrospective protection of first visible pixels and post-occlusion reappearances.

Category agnostic, bounded optical flow. Never extrapolate through a fully hidden
subject, a cut, another retained interval or an explicit user absence.
"""
from collections import OrderedDict,defaultdict
import cv2,numpy as np

REVISION=1

def covered(obj):
 return obj.get('visible',True) and obj.get('mask_allowed',True) and obj.get('presence') not in {'absent','occluded','out_of_frame'}

def key(ident):return ident[3:] if ident.startswith('m-da') else ident

class GrayFrames:
 """Decode small forward blocks once; backward walks read from bounded gray RAM."""
 def __init__(self,media):self.media=media;self.cache=OrderedDict();self.bytes=0;self.cap=None;self.decoded=0;self.colors=OrderedDict();self.color_bytes=0
 def remember_color(self,index,frame):
  if index in self.colors:self.color_bytes-=self.colors.pop(index).nbytes
  self.colors[index]=frame;self.color_bytes+=frame.nbytes
  while self.color_bytes>16*1024**2 and len(self.colors)>1:self.color_bytes-=self.colors.popitem(last=False)[1].nbytes
 def color(self,index):
  if index not in self.colors:
   if self.cap is None:self.cap=cv2.VideoCapture(str(self.media['path']))
   self.cap.set(cv2.CAP_PROP_POS_FRAMES,index)
   from video_io import frame_at
   self.remember_color(index,frame_at(self.media,index,self.cap))
  self.colors.move_to_end(index);return self.colors[index]
 def get(self,index,scale):
  cache_key=(index,scale)
  if cache_key not in self.cache:
   if self.cap is None:self.cap=cv2.VideoCapture(str(self.media['path']))
   start=index//8*8;self.cap.set(cv2.CAP_PROP_POS_FRAMES,start)
   from video_io import frame_at
   for i in range(start,min(self.media['frames'],start+8)):
    frame=frame_at(self.media,i,self.cap);gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if i==index:self.remember_color(index,frame)
    if scale!=1:gray=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    k=(i,scale)
    if k in self.cache:self.bytes-=self.cache[k].nbytes
    self.cache[k]=gray;self.bytes+=gray.nbytes;self.decoded+=1
   while self.bytes>96*1024**2 and len(self.cache)>1:
    oldest=next(iter(self.cache))
    if oldest==cache_key:self.cache.move_to_end(oldest);continue
    _,old=self.cache.popitem(last=False);self.bytes-=old.nbytes
  self.cache.move_to_end(cache_key);return self.cache[cache_key]
 def close(self):
  if self.cap is not None:self.cap.release()

def patch(gray,point,radius=3):
 size=max(5,round(radius*2+1));tile=cv2.getRectSubPix(gray,(size,size),tuple(map(float,point)))
 return cv2.resize(tile,(7,7),interpolation=cv2.INTER_AREA).astype(np.float32)

def correlation(a,b):
 if abs(float(a.mean()-b.mean()))>45:return -1.
 a=a-a.mean();b=b-b.mean();return float((a*b).sum()/max(1.,np.linalg.norm(a)*np.linalg.norm(b)))

def partial_pixels(reference,current,transform,mask,outline):
 """Verify thin visible strips after too many points have become occluded."""
 moved=cv2.transform(outline,transform).reshape(-1,2);left,top=np.floor(moved.min(axis=0)).astype(int);right,bottom=np.ceil(moved.max(axis=0)).astype(int)
 left,top=max(0,left),max(0,top);right,bottom=min(current.shape[1],right),min(current.shape[0],bottom)
 if min(right-left,bottom-top)<2:return None
 local=transform.copy();local[:,2]-=[left,top];size=(right-left,bottom-top)
 template=cv2.warpAffine(reference,local,size);allowed=cv2.warpAffine(mask,local,size,flags=cv2.INTER_NEAREST);image=current[top:bottom,left:right]
 matches=np.uint8((np.abs(template.astype(float)-image.astype(float))<=20)&(allowed>0))
 visible=np.zeros_like(current)
 # Thin strips need independent validation: random adjacent matching pixels
 # must not enlarge their component and hide the first one or two visible columns.
 height,width=image.shape;strips=[]
 for thickness in [1,2,3,4,6]:
  if thickness<=width:strips.extend([(0,0,thickness,height),(width-thickness,0,thickness,height)])
  if thickness<=height:strips.extend([(0,0,width,thickness),(0,height-thickness,width,thickness)])
 for rx,ry,rw,rh in strips:
  a=template[ry:ry+rh,rx:rx+rw].astype(np.float32);b=image[ry:ry+rh,rx:rx+rw].astype(np.float32);valid=allowed[ry:ry+rh,rx:rx+rw]>0
  if a.size<12 or valid.mean()<.9 or a.std()<8 or b.std()<8:continue
  if matches[ry:ry+rh,rx:rx+rw].mean()>=.9 and correlation(a,b)>=.94:visible[top+ry:top+ry+rh,left+rx:left+rx+rw]=255
 count,labels,stats,_=cv2.connectedComponentsWithStats(matches)
 for n in range(1,count):
  if stats[n,cv2.CC_STAT_AREA]<12 or max(stats[n,cv2.CC_STAT_WIDTH],stats[n,cv2.CC_STAT_HEIGHT])<6:continue
  member=labels==n;rx,ry,rw,rh=stats[n,:4]
  # Validate the whole bounding patch, not just pixels selected for matching:
  # the latter would falsely correlate random occluder texture by selection bias.
  if member[ry:ry+rh,rx:rx+rw].mean()<.8:continue
  a=template[ry:ry+rh,rx:rx+rw].astype(np.float32);b=image[ry:ry+rh,rx:rx+rw].astype(np.float32)
  if a.std()<8 or b.std()<8 or correlation(a,b)<.9:continue
  visible[top:bottom,left:right][member]=255
 return visible if np.count_nonzero(visible)>=12 else None

def explicit_absence(cfg,obj,index,fps):
 if cfg is None:return False
 ids={key(obj['id']),*[key(a) for a in obj.get('reference_ids',[])]};time=index/fps
 for value in cfg.confirmed_gaps:
  try:
   subject,start,end,_=value.rsplit(':',3)
   if key(subject) in ids and int(start)<=index<=int(end):return True
  except (ValueError,AttributeError):pass
 return any(key(d.subject_id) in ids and d.action in {'absent','occluded','suppress','reject'} and d.start<=time<d.end for d in cfg.temporal_decisions)

def recover(media,frames,cuts,job,cfg=None,coverage=None,checked_ids=None,frame_reader=None):
 if media['type']!='video':return {'revision':REVISION,'added_frames':0,'checked_ids':[]}
 from inspection import source_at
 from foreground_tracking import foreground
 from performance import phase,tick,timed
 coverage=coverage or [(0,len(frames))];fps=media['fps'];observations=defaultdict(dict)
 allowed=None
 if cfg is not None and cfg.library_only:
  allowed={key(s.id) for s in __import__('inspection').library_seeds(cfg)}
 excluded={key(v) for v in getattr(cfg,'excluded_ids',[])};already=set(checked_ids or [])
 for i,row in enumerate(frames):
  for obj in row:
   ids={key(obj['id']),*[key(a) for a in obj.get('reference_ids',[])]}
   if ids&excluded or allowed is not None and not ids&allowed or obj['id'] in already:continue
   observations[obj['id']][i]=obj
 manual={s.id:{round(a.time*fps) for a in [s,*s.anchors]} for s in getattr(cfg,'custom',[]) if not s.id.startswith(('a','m-da')) and not s.hidden};episodes=[];checked=[]
 for ident,rows in observations.items():
  reliable=[(i,o) for i,o in sorted(rows.items()) if (o.get('detected',False) or i in manual.get(ident,set())) and not o.get('identity_ambiguous') and (o.get('existence')!='candidate' or allowed is not None) and o.get('visible',True)]
  if not reliable:continue
  checked.append(ident)
  for index,obj in reliable:
   if index<=0 or not any(a<index<b for a,b in coverage):continue
   if index-1 in rows and covered(rows[index-1]):
    previous=rows[index-1]
    if previous.get('detected',False):continue
    # A still valid envelope that fully contains this native detection already
    # protects its appearance; avoid decoding another backward episode for it.
    x,y,w,h=obj['box'];px,py,pw,ph=previous['box'];area=max(0,min(x+w,px+pw)-max(x,px))*max(0,min(y+h,py+ph)-max(y,py))
    if not previous.get('polygons') and area/max(1,w*h)>=.98:continue
   score=obj.get('detection_confidence',obj.get('confidence',0))
   if score<(.45 if allowed is not None else .6) and not obj.get('existence_verified') and obj['id'].startswith('a'):continue
   episodes.append((index,obj))
 episodes.sort(key=lambda e:e[0])
 previous_threads=cv2.getNumThreads()
 if previous_threads>4:cv2.setNumThreads(4)
 phase('Protection des apparitions',len(episodes),'apparitions');reader=frame_reader or GrayFrames(media);added=0;partial=0
 try:
  for ordinal,(anchor,obj) in enumerate(episodes,1):
   if job.get('cancel'):raise InterruptedError('Protection des apparitions annulée.')
   box=np.array(obj['box'],np.float32);scale=min(1.,1280/max(media['width'],media['height']))
   if min(box[2:])*scale<14:scale=1.
   gray=reader.get(anchor,scale);native=reader.color(anchor)
   mask=np.zeros(gray.shape,np.uint8)
   if obj.get('polygons'):
    cv2.fillPoly(mask,[np.rint(np.array(p)*scale).astype(np.int32) for p in obj['polygons']],255)
   else:
    segment_scale=min(1.,160/max(box[2:]));segment_image=cv2.resize(native,None,fx=segment_scale,fy=segment_scale,interpolation=cv2.INTER_AREA) if segment_scale<1 else native
    with timed('tracking/onset_foreground'):separated,separation_ok=foreground(segment_image,box*segment_scale)
    if separation_ok:mask=cv2.resize(separated,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST)
    else:
     x,y,w,h=box*scale;x1,y1=max(0,round(x+w*.12)),max(0,round(y+h*.12));x2,y2=min(gray.shape[1],round(x+w*.88)),min(gray.shape[0],round(y+h*.88));mask[y1:y2,x1:x2]=255
   radius=min(3,max(1,round(float(min(box[2:])*scale)*.1)));interior=cv2.erode(mask,np.ones((radius*2+1,radius*2+1),np.uint8))
   rx,ry=max(0,int(box[0]*scale)),max(0,int(box[1]*scale));rr,rb=min(gray.shape[1],int(np.ceil((box[0]+box[2])*scale))),min(gray.shape[0],int(np.ceil((box[1]+box[3])*scale)))
   if rr-rx<3 or rb-ry<3:tick(ordinal);continue
   points=cv2.goodFeaturesToTrack(gray[ry:rb,rx:rr],maxCorners=80,qualityLevel=.01,minDistance=2,mask=interior[ry:rb,rx:rr])
   if points is not None:points+=np.array([rx,ry],np.float32)
   if points is None or len(points)<4:tick(ordinal);continue
   reference=[patch(gray,p[0],radius) for p in points];initial=len(points);minimum=max(3,min(6,round(initial*.12)));outline=np.array([[[box[0],box[1]]],[[box[0]+box[2],box[1]]],[[box[0]+box[2],box[1]+box[3]]],[[box[0],box[1]+box[3]]]],np.float32)*scale
   original_gray=gray;original_mask=mask.copy();original_outline=outline.copy();transform=np.array([[1.,0,0],[0,1.,0]],np.float32);later=[(i,o) for i,o in observations[obj['id']].items() if anchor<i<=anchor+3 and o.get('detected')];speed=np.median([(np.array(o['box'][:2])-box[:2])/(i-anchor) for i,o in later],axis=0)*scale if later else np.zeros(2);last_matrix=np.array([[1.,0,-speed[0]],[0,1.,-speed[1]]],np.float32);native_velocity=np.median([(np.array(o['box'])-box)/(i-anchor) for i,o in later],axis=0) if later else np.zeros(4);fallback_steps=0;fallback_only=False
   total_scale=1.;horizon=min(60,max(4,round(fps*(1.2 if getattr(cfg,'profile','rapide')=='extreme' else .65))))
   for index in range(anchor-1,max(-1,anchor-horizon-1),-1):
    if job.get('cancel'):raise InterruptedError('Protection des apparitions annulée.')
    if index+1 in cuts or not any(a<=index and anchor<b for a,b in coverage) or explicit_absence(cfg,obj,index,fps):break
    old=observations[obj['id']].get(index)
    if old is not None and covered(old) and old.get('detected',False):break
    current=reader.get(index,scale)
    with timed('tracking/visible_onset_flow'):
     good=None;matrix=None
     if not fallback_only:
      forward,status,_=cv2.calcOpticalFlowPyrLK(gray,current,points,None,winSize=(21,21),maxLevel=3)
      if forward is not None:
       backward,back_status,_=cv2.calcOpticalFlowPyrLK(current,gray,forward,None,winSize=(21,21),maxLevel=3)
       if backward is not None:
        keep=(status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm(backward-points,axis=2).ravel()<1.)
        for n,p in enumerate(forward):
         if keep[n] and correlation(reference[n],patch(current,p[0],radius*total_scale))<.72:keep[n]=False
        pxs,pys=forward[:,0,0],forward[:,0,1]
        for other in frames[index]:
         if other['id']==obj['id'] or other.get('category',other['class'])!=obj.get('category',obj['class']) or not covered(other) or other.get('privacy_coverage'):continue
         ox,oy,ow,oh=np.array(other['box'])*scale
         keep&=~((pxs>=ox)&(pxs<ox+ow)&(pys>=oy)&(pys<oy+oh))
        if keep.sum()>=minimum:
         proposal,inliers=cv2.estimateAffinePartial2D(points[keep],forward[keep],method=cv2.RANSAC,ransacReprojThreshold=1.5)
         if proposal is not None and inliers is not None and inliers.mean()>=.75 and .85<float(np.linalg.norm(proposal[:,0]))<1.18:
          good=np.flatnonzero(keep)[inliers.ravel()>0]
          if len(good)>=minimum:
           center=outline.mean(axis=0,keepdims=True);predicted=cv2.transform(center,last_matrix);measured=cv2.transform(center,proposal)
           expected_speed=float(np.linalg.norm(predicted-center));tolerance=max(1.5,min(4.,float(min(box[2:])*scale)*.04))+.25*expected_speed
           if float(np.linalg.norm(measured-predicted))<=tolerance:matrix=proposal
     if matrix is not None:
      moved=cv2.transform(outline,matrix)
      if float(np.linalg.norm(moved.mean(axis=0)-outline.mean(axis=0)))>max(12.,float(np.linalg.norm(box[2:]*scale))*.75):break
      outline=moved;transform=(np.vstack([matrix,[0,0,1]])@np.vstack([transform,[0,0,1]]))[:2].astype(np.float32);last_matrix=matrix;total_scale*=float(np.linalg.norm(matrix[:,0]))
      points=forward[good];reference=[reference[n] for n in good];gray=current
      support=np.zeros(current.shape,np.uint8);cv2.fillConvexPoly(support,cv2.convexHull(np.rint(points).astype(np.int32)),255)
     else:
      if fallback_steps>=horizon:break
      proposed=(np.vstack([last_matrix,[0,0,1]])@np.vstack([transform,[0,0,1]]))[:2].astype(np.float32)
      support=partial_pixels(original_gray,current,proposed,original_mask,original_outline)
      if support is None:
       # Sparse points on a thin sliver can bias scale. Cross-check the motion
       # measured in the later reliable detections, still requiring pixel evidence.
       projected=box-native_velocity*(anchor-index);sx,sy=projected[2]/box[2],projected[3]/box[3]
       if .5<sx<2 and .5<sy<2:
        native_transform=np.array([[sx,0,(projected[0]-box[0]*sx)*scale],[0,sy,(projected[1]-box[1]*sy)*scale]],np.float32)
        support=partial_pixels(original_gray,current,native_transform,original_mask,original_outline)
        if support is not None:proposed=native_transform
      if support is None:break
      transform=proposed;outline=cv2.transform(original_outline,transform);fallback_steps+=1;fallback_only=True
     x0,y0=outline.reshape(-1,2).min(axis=0);x1,y1=outline.reshape(-1,2).max(axis=0)
     if old is not None and covered(old):
      ox,oy,ow,oh=np.array(old['box'])*scale;ys,xs=np.nonzero(support)
      if len(xs) and np.mean((xs>=ox)&(xs<=ox+ow)&(ys>=oy)&(ys<=oy+oh))>=.95:break
     x,y=outline.reshape(-1,2).min(axis=0)/scale;right,bottom=outline.reshape(-1,2).max(axis=0)/scale
     x,y=max(0.,float(x)),max(0.,float(y));right,bottom=min(media['width'],float(right)),min(media['height'],float(bottom))
     if min(right-x,bottom-y)<2:break
     # Dilate the measured visible support so sparse points cannot expose skin edges.
     visible=support if scale==1 else cv2.resize(support,(media['width'],media['height']),interpolation=cv2.INTER_NEAREST)
     left,top=int(np.floor(x)),int(np.floor(y));end_x,end_y=int(np.ceil(right)),int(np.ceil(bottom));local=visible[top:end_y,left:end_x]
     pad=max(4,round(min(right-x,bottom-y)*.35));local=cv2.dilate(local,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(pad*2+1,pad*2+1)))
     contours,_=cv2.findContours(local,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);polygons=[(c.reshape(-1,2)+[left,top]).tolist() for c in contours if len(c)>=3]
     if not polygons:break
     found=dict(obj,box=[x,y,right-x,bottom-y],polygons=polygons,detected=False,visible=True,mask_allowed=True,presence='visible',existence=obj.get('existence','confirmed'),identity_ambiguous=False,privacy_coverage=True,future_confirmed=True,visible_onset_guard=True,onset_method='partial-texture' if fallback_only else 'flow',tracking_confidence=.8,detection_confidence=0.,timestamp=index/fps)
     if old is not None:frames[index].remove(old)
     frames[index].append(found);observations[obj['id']][index]=found;added+=1
     if fallback_only or len(good)<initial*.7:partial+=1
   tick(ordinal)
 finally:
  if frame_reader is None:reader.close()
  if previous_threads>4:cv2.setNumThreads(previous_threads)
 stats={'revision':REVISION,'added_frames':added,'partial_visibility_frames':partial,'episodes':len(episodes),'checked_ids':checked,'decoded_frames':getattr(reader,'decoded',0)}
 job['visible_onsets']=stats
 return stats



