"""Short, conservative entry guards from measured inward motion at a frame edge."""
from collections import defaultdict
from statistics import median
import math
from videomask.persons import iou

def extend(media,frames,cuts,first=0,last=None,coverage=None):
 last=last or len(frames);coverage=coverage or [(first,last)];tracks=defaultdict(list);added=0
 for i in range(first,last):
  for obj in frames[i]:
   if obj.get('visible',True) and obj.get('presence')!='absent' and obj.get('category',obj.get('class')) in {'personne','tête','visage'}:tracks[obj['id']].append((i,obj))
 for ident,rows in tracks.items():
  anchor,obj=rows[0]
  if anchor<=first or len(rows)<3:continue
  neighbors=[(i,o) for i,o in rows[:6] if i-anchor<=max(4,round(media['fps']*.3))]
  speeds=[]
  for (a,left),(b,right) in zip(neighbors,neighbors[1:]):
   if b<=a:continue
   x,y,w,h=left['box'];xx,yy,ww,hh=right['box'];speeds.append(((xx+ww/2-x-w/2)/(b-a),(yy+hh/2-y-h/2)/(b-a)))
  if len(speeds)<2:continue
  vx,vy=median(v[0] for v in speeds),median(v[1] for v in speeds);x,y,w,h=obj['box'];mw,mh=media['width'],media['height']
  inward=(x<=mw*.04 and vx>.25) or (x+w>=mw*.96 and vx<-.25) or (y<=mh*.04 and vy>.25) or (y+h>=mh*.96 and vy<-.25)
  if not inward or math.hypot(vx,vy)>.35*math.hypot(w,h):continue
  for step in range(1,min(60,max(4,round(media['fps']*2)))+1):
   i=anchor-step
   if i<first or any(i<c<=anchor for c in cuts) or not any(a<=i and anchor<b for a,b in coverage):break
   if any(o['id']==ident for o in frames[i]):break
   nx,ny=x-vx*step,y-vy*step;xa,ya=max(0,nx),max(0,ny);xb,yb=min(mw,nx+w),min(mh,ny+h)
   if min(xb-xa,yb-ya)<2:break
   box=[xa,ya,xb-xa,yb-ya]
   if any(o['id']!=ident and o.get('category',o.get('class'))==obj.get('category',obj.get('class')) and iou(tuple(box),tuple(o['box']))>.5 for o in frames[i]):break
   guard=dict(obj,box=box,detected=False,visible=True,mask_allowed=True,presence='visible',privacy_coverage=True,edge_entry_guard=True,future_confirmed=True,tracking_confidence=.55)
   guard.pop('polygons',None);guard.pop('manual_contour',None)
   frames[i].append(guard);added+=1
 return added
