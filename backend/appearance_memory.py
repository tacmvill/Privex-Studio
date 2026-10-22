"""Optional backward visual memory for delayed/blurred detections, entirely local.

No new identity or recognition model: color layout, low-frequency structure,
gradient shape, local motion and competing observations validate every step.
"""
from collections import OrderedDict,defaultdict
import math,time
import cv2,numpy as np
from videomask.persons import iou
from visible_onsets import explicit_absence,covered,key

REVISION=1


class Frames:
 def __init__(self,media,job):
  self.media=media;self.job=job;self.cache=OrderedDict();self.bytes=0;self.cap=None;self.decoded=0
  self.scale=min(1.,1280/max(media['width'],media['height']))
 def get(self,index):
  if index not in self.cache:
   from video_io import frame_at
   if self.cap is None:self.cap=cv2.VideoCapture(str(self.media['path']))
   start=index//16*16;self.cap.set(cv2.CAP_PROP_POS_FRAMES,start)
   for i in range(start,min(self.media['frames'],start+16)):
    if self.job.get('cancel'):raise InterruptedError('Recherche par apparence annulée.')
    frame=frame_at(self.media,i,self.cap)
    if self.scale<1:frame=cv2.resize(frame,None,fx=self.scale,fy=self.scale,interpolation=cv2.INTER_AREA)
    if i in self.cache:self.bytes-=self.cache.pop(i).nbytes
    self.cache[i]=frame;self.bytes+=frame.nbytes;self.decoded+=1
   while self.bytes>96*1024**2 and len(self.cache)>1:
    oldest=next(iter(self.cache))
    if oldest==index:self.cache.move_to_end(oldest);continue
    self.bytes-=self.cache.popitem(last=False)[1].nbytes
  self.cache.move_to_end(index);return self.cache[index]
 def close(self):
  if self.cap is not None:self.cap.release()


def crop(image,box):
 x,y,w,h=map(float,box);height,width=image.shape[:2]
 if min(w,h)<6 or x<0 or y<0 or x+w>width or y+h>height:return None
 tile=cv2.getRectSubPix(image,(max(6,round(w)),max(6,round(h))),(x+w/2,y+h/2))
 return cv2.resize(tile,(48,48),interpolation=cv2.INTER_AREA)


def describe(tile):
 colors=[];layouts=[]
 for sigma in [0.,2.,4.]:
  image=cv2.GaussianBlur(tile,(0,0),sigma) if sigma else tile
  hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);hist=cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256]).ravel();hist/=max(1e-6,np.linalg.norm(hist));colors.append(hist)
  lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32);layouts.append(cv2.resize(lab,(4,4),interpolation=cv2.INTER_AREA))
 gray=cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY).astype(np.float32);low=[]
 for sigma in [0.,1.6,3.2]:
  small=cv2.resize(cv2.GaussianBlur(gray,(0,0),sigma) if sigma else gray,(12,12),interpolation=cv2.INTER_AREA);small-=small.mean();small/=max(1e-6,np.linalg.norm(small));low.append(small.ravel())
 gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1);magnitude,angle=cv2.cartToPolar(gx,gy);bins=np.minimum(7,(angle*8/(2*math.pi)).astype(int));edges=np.bincount(bins.ravel(),weights=magnitude.ravel(),minlength=8).astype(np.float32);edges/=max(1e-6,np.linalg.norm(edges))
 return {'tile':tile,'colors':colors,'layouts':layouts,'low':low,'edges':edges,'focus':float(cv2.Laplacian(gray,cv2.CV_32F).var()),'contrast':float(gray.std())}


def compare(a,b):
 color,layout=max([(float(x@y),math.exp(-float(np.abs(xx-yy).mean())/22.)) for x,xx in zip(a['colors'],a['layouts']) for y,yy in zip(b['colors'],b['layouts'])],key=lambda pair:pair[0]+pair[1])
 structure=max(float(x@y) for x in a['low'] for y in b['low']);edges=float(a['edges']@b['edges'])
 blurred=min(a['focus'],b['focus'])<max(a['focus'],b['focus'])*.35
 score=.25*color+(.35 if blurred else .25)*layout+.35*max(0.,structure)+(.05 if blurred else .15)*edges
 return score,color,layout,structure


def locate(image,expected,recent,gallery):
 """Bounded local template hypotheses; attributes and a second peak veto aliases."""
 x,y,w,h=expected;height,width=image.shape[:2];radius=max(5.,min(max(w,h)*.8,80.))
 left,top=max(0,math.floor(x-radius)),max(0,math.floor(y-radius));right,bottom=min(width,math.ceil(x+w+radius)),min(height,math.ceil(y+h+radius))
 region=image[top:bottom,left:right]
 if not region.size:return None
 gray=cv2.GaussianBlur(cv2.cvtColor(region,cv2.COLOR_BGR2GRAY),(0,0),1.2);candidates=[]
 for factor in [1.,.96,1.04]:
  tw,th=max(6,round(w*factor)),max(6,round(h*factor))
  if tw>region.shape[1] or th>region.shape[0]:continue
  for view in [recent,*[cv2.GaussianBlur(gallery[0]['tile'],(0,0),sigma) for sigma in [2.,4.]]]:
   tile=cv2.resize(view,(tw,th));template=cv2.GaussianBlur(cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY),(0,0),1.2)
   if template.std()<2.5:continue
   scores=cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED);scores=np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
   for _ in range(2):
    _,peak,_,pos=cv2.minMaxLoc(scores)
    if peak<.35:break
    px,py=pos;box=[float(left+px),float(top+py),float(tw),float(th)];patch=crop(image,box)
    if patch is None:break
    descriptor=describe(patch);attributes=max([compare(descriptor,v) for v in gallery],key=lambda v:v[0]);score,color,layout,structure=attributes
    displacement=math.hypot(box[0]+tw/2-x-w/2,box[1]+th/2-y-h/2)/max(8,math.hypot(w,h))
    valid=descriptor['contrast']>=3 and color>=.7 and layout>=.6 and structure>=.45 and score>=.76
    if valid:candidates.append((score+.05*math.exp(-displacement)-.03*abs(math.log(factor)),score,box,patch,descriptor))
    scores[max(0,py-th//2):min(scores.shape[0],py+th//2+1),max(0,px-tw//2):min(scores.shape[1],px+tw//2+1)]=-1
 candidates.sort(key=lambda c:c[0],reverse=True)
 if not candidates:return None
 best=candidates[0];competitors=[c for c in candidates[1:] if iou(tuple(c[2]),tuple(best[2]))<.35]
 if competitors and best[0]-competitors[0][0]<.045:return None
 return best[1:]


def recover(media,frames,cuts,job,cfg,coverage=None,reader=None):
 stats={'revision':REVISION,'enabled':bool(getattr(cfg,'appearance_memory',False)),'added_frames':0,'subjects_extended':0,'episodes':0,'decoded_frames':0,'budget_limited':False}
 if not stats['enabled'] or media['type']!='video':return stats
 from performance import phase,tick,timed
 coverage=coverage or [(0,len(frames))];fps=media['fps'];tracks=defaultdict(dict);excluded={key(v) for v in getattr(cfg,'excluded_ids',[])}
 for i,row in enumerate(frames):
  for obj in row:
   if {key(obj['id']),*[key(a) for a in obj.get('reference_ids',[])]}&excluded:continue
   tracks[obj['id']][i]=obj
 episodes=[]
 for ident,rows in tracks.items():
  native=[(i,o) for i,o in sorted(rows.items()) if o.get('detected') and o.get('visible',True) and not o.get('identity_ambiguous') and (o.get('existence')=='confirmed' or o.get('existence_verified'))]
  if not native:continue
  for n,(i,o) in enumerate(native):
   # A later clear observation may validate earlier blurred pixels of this ID.
   if n and i-native[n-1][0]<=max(2,round(fps*.2)) and not any(native[n-1][0]<c<=i for c in cuts):continue
   if not any(a<i<b for a,b in coverage):continue
   if o.get('detection_confidence',o.get('confidence',0))<.5 and not o.get('existence_verified'):continue
   episodes.append((i,ident,o,[(at,v) for at,v in native[n:n+5] if not any(i<c<=at for c in cuts) and any(a<=i and at<b for a,b in coverage)]))
 phase('Recherche par apparence',len(episodes),'apparitions');reader=reader or Frames(media,job);scale=getattr(reader,'scale',1.);start=time.monotonic();budget=30 if cfg.profile=='extreme' else 18;extended=set()
 try:
  for ordinal,(anchor,ident,obj,following) in enumerate(sorted(episodes),1):
   if job.get('cancel'):raise InterruptedError('Recherche par apparence annulée.')
   if time.monotonic()-start>budget:stats['budget_limited']=True;break
   stats['episodes']+=1;box=np.asarray(obj['box'],float)*scale;image=reader.get(anchor);recent=crop(image,box)
   if recent is None:tick(ordinal);continue
   reference=describe(recent);gallery=[reference]
   for i,other in sorted(following[1:],key=lambda v:v[1].get('detection_confidence',0),reverse=True)[:2]:
    if i-anchor>fps*2:continue
    tile=crop(reader.get(i),np.asarray(other['box'])*scale)
    if tile is not None:gallery.append(describe(tile))
   deltas=[(np.asarray(o['box'],float)-np.asarray(obj['box']))*scale/(i-anchor) for i,o in following[1:] if i>anchor]
   velocity=np.median(deltas,axis=0) if deltas else np.zeros(4);size_velocity=velocity[2:].copy();native_size=box[2:].copy();last_box=box.copy()
   for index in range(anchor-1,max(-1,anchor-round(fps*5)-1),-1):
    if job.get('cancel'):raise InterruptedError('Recherche par apparence annulée.')
    if time.monotonic()-start>budget:stats['budget_limited']=True;break
    if index+1 in cuts or not any(a<=index and anchor<b for a,b in coverage) or explicit_absence(cfg,obj,index,fps):break
    old=tracks[ident].get(index)
    if old and old.get('presence') in {'absent','out_of_frame','occluded'}:break
    if old and covered(old):
     if old.get('detected'):break
     last_box=np.asarray(old['box'],float)*scale;recent=crop(reader.get(index),last_box)
     if recent is None:break
     continue
    expected=last_box-velocity
    expected[2:]=native_size-size_velocity*(anchor-index)
    if min(expected[2:])<6 or np.any(expected[2:]<native_size*.2) or np.any(expected[2:]>native_size*2.5):break
    current=reader.get(index)
    with timed('tracking/appearance_memory'):found=locate(current,expected,recent,gallery)
    if found is None:break
    score,located,tile,descriptor=found;located=np.asarray(located,float)
    # Another observed subject of this type is a competing identity, not a match.
    if any(other['id']!=ident and other.get('category',other['class'])==obj.get('category',obj['class']) and covered(other) and iou(tuple(np.asarray(other['box'])*scale),tuple(located))>.35 for other in frames[index]):break
    measured=last_box-located;velocity=.7*velocity+.3*measured;velocity[2:]=size_velocity;last_box=located;recent=tile
    # Keep native gallery immutable: uncertain blurred crops cannot learn scenery.
    # Small, bounded coverage margin: head memory never turns into a torso mask.
    native=located/scale;x,y,w,h=native;pad=.08;left=max(0,x-w*pad);top=max(0,y-h*pad);right=min(media['width'],x+w*(1+pad));bottom=min(media['height'],y+h*(1+pad))
    found=dict(obj,box=[round(left),round(top),round(right-left),round(bottom-top)],detected=False,detection_confidence=0.,visible=True,mask_allowed=True,presence='visible',identity_ambiguous=False,tracking_confidence=min(.95,score),appearance_confidence=score,appearance_memory_guard=True,privacy_coverage=True,tracking_status='visual_memory',timestamp=index/fps)
    found.pop('polygons',None);found.pop('manual_contour',None)
    if old:frames[index].remove(old)
    frames[index].append(found);tracks[ident][index]=found;stats['added_frames']+=1;extended.add(ident)
   tick(ordinal)
 finally:
  stats['decoded_frames']=getattr(reader,'decoded',0);reader.close()
 # Report the same recovered lifetime in the inspector as in the rendered plan.
 from temporal_intelligence import spans
 for trajectory in job.get('trajectories',[]):
  if trajectory['id'] not in extended:continue
  states={i:span['state'] for span in trajectory.get('intervals',[]) for i in range(span['first'],span['last']+1)}
  recovered=[i for i,obj in tracks[trajectory['id']].items() if obj.get('appearance_memory_guard')]
  states.update({i:'visible' for i in recovered});trajectory['intervals']=spans(sorted(states.items()),fps);trajectory['appearance_memory_frames']=len(recovered)
 stats['subjects_extended']=len(extended);stats['seconds']=round(time.monotonic()-start,3)
 if stats['budget_limited']:job.setdefault('warnings',[]).append('Recherche par apparence : certaines apparitions n’ont pas été prolongées dans le budget de cette passe.')
 job['appearance_memory']=stats;return stats
