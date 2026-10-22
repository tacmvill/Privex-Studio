"""Conservative rendering continuity for one/two-frame holes in an existing ID.

No new identity, extrapolation, cross-shot link or inference is introduced.
Explicit suppression decisions are still resolved after this pass.
"""
import math
from collections import defaultdict

def repair(plan):
 if plan.get('render_continuity_version')==3:return 0
 frames=plan['frames'];cuts=plan.get('scene_cuts',[]);coverage=plan.get('analysis',{}).get('coverage',[(0,len(frames))]);tracks=defaultdict(list);count=0
 for i,row in enumerate(frames):
  for obj in row:
   if obj.get('visible',True) and obj.get('presence') not in {'absent','occluded'} and (obj.get('detected') or obj.get('tracking_confidence',1)>=.5):tracks[obj['id']].append((i,obj))
 for ident,observed in tracks.items():
  for (a,left),(b,right) in zip(observed,observed[1:]):
   if not 1<b-a<=3 or any(a<c<=b for c in cuts) or not any(start<=a and b<end for start,end in coverage):continue
   if left.get('category',left.get('class'))!=right.get('category',right.get('class')):continue
   x,y,w,h=left['box'];xx,yy,ww,hh=right['box']
   if min(w,h,ww,hh)<=0 or not .5<=ww*hh/(w*h)<=2:continue
   if math.hypot(xx+ww/2-x-w/2,yy+hh/2-y-h/2)>.65*max(math.hypot(w,h),math.hypot(ww,hh)):continue
   # Verified absence blocks the entire short bridge, even if it lasts one frame.
   if any(o['id']==ident and o.get('presence')=='absent' for row in frames[a+1:b] for o in row):continue
   for i in range(a+1,b):
    old=next((o for o in frames[i] if o['id']==ident),None)
    if old and old.get('visible',True) and old.get('presence')!='occluded' and old.get('mask_allowed',True):continue
    t=(i-a)/(b-a);box=[v+(z-v)*t for v,z in zip(left['box'],right['box'])]
    obj=dict(left,box=box,visible=True,mask_allowed=True,detected=False,render_gap_bridge=True,tracking_status='interpolated')
    anchor=left if left.get('polygons') else right
    if anchor.get('polygons'):
     ox,oy,ow,oh=anchor['box'];nx,ny,nw,nh=box
     obj['polygons']=[[[nx+(px-ox)*nw/ow,ny+(py-oy)*nh/oh] for px,py in polygon] for polygon in anchor['polygons']]
    if old:frames[i].remove(old)
    frames[i].append(obj);count+=1
 # A one-frame collapsed/misplaced envelope must not uncover a stable subject.
 from videomask.persons import iou
 for ident,observed in tracks.items():
  for (a,left),(i,current),(b,right) in zip(observed,observed[1:],observed[2:]):
   if i!=a+1 or b!=i+1 or any(a<c<=b for c in cuts):continue
   if not any(start<=a and b<end for start,end in coverage):continue
   if iou(tuple(left['box']),tuple(right['box']))<.5:continue
   box=current['box'];predicted=[(x+y)/2 for x,y in zip(left['box'],right['box'])]
   contracts=box[1]>predicted[1]+predicted[3]*.06 or box[1]+box[3]<predicted[1]+predicted[3]*.92
   if not contracts and box[2]*box[3]>=predicted[2]*predicted[3]*.65 and iou(tuple(box),tuple(predicted))>=.3:continue
   x,y=min(box[0],predicted[0]),min(box[1],predicted[1]);r=max(box[0]+box[2],predicted[0]+predicted[2]);bottom=max(box[1]+box[3],predicted[1]+predicted[3])
   current.update(box=[x,y,r-x,bottom-y],privacy_coverage=True,render_envelope_guard=True)
   current.pop('polygons',None);count+=1
 plan['render_continuity_version']=3;plan['render_gap_bridged_frames']=count
 return count
