"""Reversible linear mask bridges between two observations of one identity.

No decoding, recognition or identity association; interpolation is an optional
rendering estimate, not evidence that the subject was visible behind an obstacle.
"""
from collections import defaultdict
import math

def selected(obj,cfg=None):
 if not obj.get('linear_gap_bridge'):return obj
 return obj if getattr(cfg,'linear_tracking',False) else obj.get('linear_original')

def prepare(plan,fps):
 if plan.get('linear_tracking_revision')==1:return 0
 frames=plan['frames'];cuts=plan.get('scene_cuts',[]);coverage=plan.get('analysis',{}).get('coverage',[(0,len(frames))]);tracks=defaultdict(list);added=0
 for i,row in enumerate(frames):
  for obj in row:
   if obj.get('linear_gap_bridge'):continue
   if obj.get('visible',True) and obj.get('mask_allowed',True) and obj.get('presence') not in {'absent','occluded','out_of_frame'} and not obj.get('identity_ambiguous') and obj.get('existence')!='candidate' and (obj.get('detected') or obj.get('visible_onset_guard') or obj.get('tracking_confidence',0)>=.65):tracks[obj['id']].append((i,obj))
 for ident,observed in tracks.items():
  for (a,left),(b,right) in zip(observed,observed[1:]):
   if not 1<b-a<=max(2,round(fps)) or any(a<c<=b for c in cuts) or not any(start<=a and b<end for start,end in coverage):continue
   if left.get('category',left.get('class'))!=right.get('category',right.get('class')):continue
   x,y,w,h=left['box'];xx,yy,ww,hh=right['box']
   if min(w,h,ww,hh)<=0 or not .25<=ww*hh/(w*h)<=4:continue
   if math.hypot(xx+ww/2-x-w/2,yy+hh/2-y-h/2)>4*max(math.hypot(w,h),math.hypot(ww,hh)):continue
   # Out-of-frame/verified absence must not become a made-up trajectory.
   if any(o['id']==ident and o.get('presence') in {'absent','out_of_frame'} for row in frames[a+1:b] for o in row):continue
   for i in range(a+1,b):
    old=next((o for o in frames[i] if o['id']==ident),None)
    if old and old.get('visible',True) and old.get('mask_allowed',True) and old.get('presence') not in {'occluded','doubt'}:continue
    t=(i-a)/(b-a);box=[v+(z-v)*t for v,z in zip(left['box'],right['box'])]
    # Keep a small coverage guard around the estimated path for partial faces.
    nx,ny,nw,nh=box;pad=.08;box=[nx-nw*pad,ny-nh*pad,nw*(1+2*pad),nh*(1+2*pad)]
    obj=dict(left,box=box,visible=True,mask_allowed=True,presence='visible',detected=False,identity_ambiguous=False,linear_gap_bridge=True,linear_original=old,tracking_status='linear_estimate',tracking_confidence=.5,privacy_coverage=True,timestamp=i/fps)
    # The uncertain interval uses the full envelope. Reusing a segmented face
    # fragment here would leave the unseen part uncovered on reappearance.
    obj.pop('polygons',None);obj.pop('manual_contour',None)
    if old:frames[i].remove(old)
    frames[i].append(obj);added+=1
 plan['linear_tracking_revision']=1;plan['linear_tracking_frames']=added
 return added
