"""Conservative scene-level absence evidence; an occlusion is never an absence by default."""
import cv2
import numpy as np

HUMAN={'personne','tête','visage'}

def edge_motion(media,observations,index,departure):
 """Require an actual boundary contact and coherent movement, not a missed box."""
 rows=sorted((i,o['box']) for i,o in observations.items() if o.get('visible') and o.get('tracking_confidence',0)>=.6 and abs(i-index)<=media['fps'])
 rows=[r for r in rows if r[0]<=index] if departure else [r for r in rows if r[0]>=index]
 rows=rows[-4:] if departure else rows[:4]
 if len(rows)<3:return False
 box=observations[index]['box'];x,y,w,h=box;mw,mh=media['width'],media['height']
 distances=[x,mw-x-w,y,mh-y-h];side=int(np.argmin(distances))
 if distances[side]>2:return False
 axis=0 if side<2 else 1;centres=[b[axis]+b[axis+2]/2 for _,b in rows]
 sign=(-1 if side in {0,2} else 1)*(1 if departure else -1)
 changes=np.diff(centres)*sign
 return bool((centres[-1]-centres[0])*sign>=max(3,box[axis+2]*.04) and np.all(changes>=-1))

class PresenceEvidence:
 def __init__(self,media,detector,first,last,job):
  self.media=media;self.detector=detector;self.first=first;self.last=last;self.job=job
  self.cuts=[];self.previous=None;self.observed={};self.absences=[];self.bodies={};self.subjects={}
 def observe(self,index,frame):
  small=cv2.resize(frame,(96,54),interpolation=cv2.INTER_AREA)
  cut=False
  if self.previous is not None:
   difference=float(np.abs(small.astype(np.float32)-self.previous.astype(np.float32)).mean())
   # Both appearance distribution and spatial arrangement must change substantially.
   def histogram(image):
    hist=cv2.calcHist([image],[0,1,2],None,[8,8,8],[0,256]*3);return cv2.normalize(hist,hist).flatten()
   distance=cv2.compareHist(histogram(small),histogram(self.previous),cv2.HISTCMP_BHATTACHARYYA)
   cut=difference>35 and distance>.55
  self.previous=small
  if cut:self.cuts=sorted({*self.cuts,index})
  return cut
 def record(self,index,objects):
  self.subjects[index]=[o for o in objects if o.get('auxiliary')]
  self.observed[index]={o.get('category',o['class']).casefold() for o in objects}
  self.bodies[index]=[o for o in objects if o.get('category',o['class'])=='personne' and o.get('confidence',0)>=.4]
  # Head-only detection still retains auxiliary bodies as temporal evidence;
  # they never become selectable Personne subjects or exported masks by themselves.
  from videomask.persons import iou
  for obj in objects:
   body=obj.get('body_evidence')
   if body and body.get('confidence',0)>=.4 and not any(iou(tuple(body['box']),tuple(o['box']))>.7 for o in self.bodies[index]):
    self.bodies[index].append(dict(body,category='personne',**{'class':'personne'}))
 def empty(self,start,end,category):
  from inspection import source_at
  compatible=HUMAN if category in HUMAN else {category}
  coverage=getattr(self.detector.cfg,'analysis_ranges',[])
  requested=[i for i in range(start,end) if not coverage or any(a<=i<b for a,b in coverage)]
  # Check sampled detections first; never accept a weak/filtered detection as absence.
  if any(self.observed.get(i,set()) & compatible for i in requested):return False
  self.job['phase']='Vérification des introductions et sorties de champ'
  cap=cv2.VideoCapture(str(self.media['path']));next_frame=None
  try:
   for i in requested:
    if self.job['cancel']:raise InterruptedError('Analyse annulée.')
    if i not in self.observed:
     if next_frame!=i:cap.set(cv2.CAP_PROP_POS_FRAMES,i)
     frame=__import__('video_io').frame_at(self.media,i,cap)
     next_frame=i+1;self.record(i,self.detector.detect(frame))
    if self.observed[i] & compatible:return False
  finally:cap.release()
  return True
 def absent_frames(self,seed,anchors,proposals):
  category=(seed.category or seed.label).casefold()
  # Arbitrary manual zones have no semantic detector able to prove their absence.
  if category=='custom':return set()
  supported=set(self.detector.cfg.classes)|set(self.detector.cfg.extended)
  if self.detector.cfg.heads or self.detector.cfg.faces:supported |= HUMAN
  if self.detector.cfg.ocr:supported |= {'texte','heure lisible','texte de panneau'}
  if self.detector.cfg.plates:supported.add('plaque')
  if category not in supported:return set()
  bounds=[self.first,*self.cuts,self.last];candidates=[]
  for a,b in zip(bounds,bounds[1:]):
   # A whole shot without an anchor, separated from the selected subject by a cut.
   if not any(a<=anchor<b for anchor in anchors) and (a in self.cuts or b in self.cuts):candidates.append((a,b,'Plan sans sujet'))
  reliable=sorted(i for i,options in proposals.items() if self.first<=i<self.last and any(not p[2] for p in options))
  if reliable:
   # A continuous entry/exit at the image edge also explains a negative prefix/suffix.
   for index,a,b in [(reliable[0],self.first,reliable[0]),(reliable[-1],reliable[-1]+1,self.last)]:
    box=min(proposals[index],key=lambda p:p[0])[1];x,y,w,h=box;mw=self.media['width'];mh=self.media['height']
    edge=min(x/mw,y/mh,(mw-x-w)/mw,(mh-y-h)/mh)<.035
    if edge and b-a>=max(2,round(self.media['fps']*.3)) and not any(a<=v<b for v in anchors):candidates.append((a,b,'Entrée ou sortie de champ'))
  absent=set()
  for a,b,reason in candidates:
   if self.empty(a,b,category):
    absent.update(range(a,b));self.absences.append({'subject_id':seed.id,'start':a/self.media['fps'],'end':b/self.media['fps'],'reason':reason})
  return absent

 def explained_edges(self,seed,observations,raw):
  from videomask.persons import iou
  reliable=sorted(i for i,o in observations.items() if self.first<=i<self.last and o.get('visible') and o.get('tracking_confidence',0)>=.6)
  if not reliable:return set()
  cat=(seed.category or seed.label).casefold();compatible={'visage','tête'} if cat in {'visage','tête'} else {cat};absent=set()
  if cat in {'custom',''} or not any(categories & compatible for categories in self.observed.values()):return absent
  for index,departure in [(reliable[0],False),(reliable[-1],True)]:
   if not edge_motion(self.media,observations,index,departure):continue
   neighbours=[i for i in reliable if abs(i-index)<=self.media['fps'] and (i<=index if departure else i>=index)]
   others={o['id'] for i in neighbours for o in raw[i] if o.get('visible',True) and o.get('tracking_confidence',1)>=.6 and (o.get('category') or o['class']).casefold() in compatible and iou(tuple(o['box']),tuple(observations[i]['box']))<.1}
   # Existing, simultaneously visible people cannot be this departing subject.
   # A newly appearing compatible identity remains an actionable ambiguity.
   scan=range(index+1,self.last) if departure else range(index-1,self.first-1,-1);indices=[]
   for i in scan:
    if any(o.get('visible',True) and o.get('tracking_confidence',1)>=.6 and (o.get('category') or o['class']).casefold() in compatible and o['id'] not in others for o in raw[i]):break
    indices.append(i)
   if indices:
    start,end=min(indices),max(indices)+1;absent.update(indices);self.absences.append({'subject_id':seed.id,'start':start/self.media['fps'],'end':end/self.media['fps'],'reason':'Sortie ou entrée de champ vérifiée'})
  return absent
