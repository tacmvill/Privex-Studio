"""Bounded backward search from a reliable small human, using measured foreground motion."""
import time
from collections import defaultdict
from videomask.persons import iou

def recover(media,frames,cuts,job,presence=None,first=0,last=None):
 from inspection import source_at
 from foreground_tracking import ForegroundFollower,foreground
 import cv2,numpy as np
 from temporal_intelligence import category,compatible
 from performance import timed
 last=last or len(frames);tracks=defaultdict(list)
 for i in range(first,last):
  for obj in frames[i]:
   if obj.get('detected',True) and category(obj) in {'tête','visage','personne'}:tracks[obj['id']].append((i,obj))
 budget=20 if getattr(getattr(presence,'detector',None),'cfg',None) and presence.detector.cfg.profile=='extreme' else 8
 started=time.monotonic();added=0;attempted=0
 def priority(item):
  rows=item[1];initial=max(1,rows[0][1]['box'][2]*rows[0][1]['box'][3]);growth=max(o['box'][2]*o['box'][3] for _,o in rows[:max(3,round(media['fps']*3))])/initial
  return (-growth,rows[0][0])
 for ident,rows in sorted(tracks.items(),key=priority):
  anchor,obj=rows[0];box=obj['box']
  if anchor<=first or len(rows)<3 or max(o.get('detection_confidence',o.get('confidence',0)) for _,o in rows)<.65:continue
  if time.monotonic()-started>budget:break
  if job.get('cancel'):raise InterruptedError('Recherche rétrospective annulée.')
  attempted+=1;follower=ForegroundFollower(source_at(media,anchor),box);seen=[]
  large=box[2]*box[3]>media['width']*media['height']*.025
  horizon=1.2 if large else 12 if budget==20 else 8
  for i in range(anchor-1,max(first-1,anchor-round(media['fps']*horizon)-1),-1):
   if i+1 in cuts or time.monotonic()-started>budget:break
   if job.get('cancel'):raise InterruptedError('Recherche rétrospective annulée.')
   if presence and getattr(presence.detector.cfg,'analysis_ranges',[]) and not any(a<=i<b for a,b in presence.detector.cfg.analysis_ranges):break
   image=source_at(media,i)
   with timed('tracking/retrospective_foreground'):tracked=follower.update(image)
   if tracked is None or follower.confidence<.7:break
   if (anchor-i)%max(1,round(media['fps']*.25))==0:
    x,y,w,h=tracked;context=(max(0,x-w*.35),max(0,y-h*.35),w*1.7,h*1.7);mask,separated=foreground(image,context)
    if separated:
     located=cv2.boundingRect(cv2.findNonZero(mask));similarity=float(follower.feature@follower._feature(image,mask))
     if similarity<.85 or iou(tuple(located),tuple(tracked))<.25:break
     follower=ForegroundFollower(image,located);tracked=located
   competing=[o for o in frames[i] if o['id']!=ident and compatible(category(o),category(obj)) and iou(tuple(o['box']),tuple(tracked))>.3]
   # Existing trajectories remain separate. A crossing is not proof of identity.
   if competing:break
   if any(o['id']==ident for o in frames[i]):break
   x,y,w,h=tracked
   xa,ya=max(0,x),max(0,y);xb,yb=min(media['width'],x+w),min(media['height'],y+h)
   if min(xb-xa,yb-ya)<3:break
   tracked=(xa,ya,xb-xa,yb-ya)
   seen.append((i,dict(obj,box=list(tracked),detected=False,visible=True,tracking_confidence=follower.confidence,detection_confidence=0.,future_confirmed=True,existence_verified=True,polygons=follower.polygons)))
  for i,found in seen:frames[i].append(found);added+=1
 job['retrospective_frames']=added;job['retrospective_subjects_attempted']=attempted;job['retrospective_budget_seconds']=budget
 job['edge_entry_guard_frames']=__import__('temporal_entry').extend(media,frames,cuts,first,last)
 return added
