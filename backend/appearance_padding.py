"""Short conservative onset envelopes when partial visibility defeats optical flow.

This is a rendering safety margin, not a detector or a new identity. A few frames
of foreground/scenery may be masked just before an observed subject reappears.
"""
import math
from collections import defaultdict
from visible_onsets import explicit_absence,covered

REVISION=1

def prepare(plan,media,cfg=None):
 if media['type']!='video' or plan.get('appearance_padding_revision')==REVISION:return 0
 frames=plan['frames'];fps=media['fps'];cuts=plan.get('scene_cuts',[]);coverage=plan.get('analysis',{}).get('coverage',[(0,len(frames))]);observations=defaultdict(dict);added=0
 # Rebuild optional interpolation after the automatic guard. Restoring its
 # original rows prevents an optional estimate being treated as observed data.
 if plan.get('linear_tracking_revision'):
  for i,row in enumerate(frames):
   frames[i]=[original for obj in row if (original:=obj.get('linear_original') if obj.get('linear_gap_bridge') else obj) is not None]
  plan.pop('linear_tracking_revision',None)
 for i,row in enumerate(frames):
  for obj in row:observations[obj['id']][i]=obj
 span=max(1,min(6,math.ceil(fps*.1)))
 for ident,rows in observations.items():
  reliable=[(i,obj) for i,obj in sorted(rows.items()) if obj.get('detected') and covered(obj) and obj.get('existence')!='candidate' and not obj.get('identity_ambiguous') and obj.get('detection_confidence',1)>=.45]
  for ordinal,(anchor,obj) in enumerate(reliable):
   previous=rows.get(anchor-1)
   if previous and previous.get('detected') and covered(previous):continue
   x,y,w,h=obj['box']
   if min(w,h)<2:continue
   future=[(i,o) for i,o in reliable[ordinal+1:ordinal+4] if anchor<i<=anchor+3 and not any(anchor<c<=i for c in cuts)]
   velocities=[[(v-u)/(i-anchor) for u,v in zip(obj['box'],o['box'])] for i,o in future]
   # Median motion from subsequent native observations. Never borrow motion
   # from another subject or unboundedly extrapolate a noisy box.
   if velocities:
    velocity=[sorted(v[k] for v in velocities)[len(velocities)//2] for k in range(4)]
   else:velocity=[0.,0.,0.,0.]
   velocity=[max(-limit,min(limit,v)) for v,limit in zip(velocity,[w*.25,h*.25,w*.08,h*.08])]
   for index in range(anchor-1,max(-1,anchor-span-1),-1):
    if index+1 in cuts or not any(a<=index and anchor<b for a,b in coverage) or explicit_absence(cfg,obj,index,fps):break
    old=rows.get(index)
    if old and old.get('detected') and covered(old):break
    if old and old.get('appearance_padding'):break
    dt=anchor-index;nx,ny,nw,nh=[v-dt*speed for v,speed in zip(obj['box'],velocity)]
    pad_x=max(3,nw*.12);pad_y=max(3,nh*.12)
    left,top=max(0.,nx-pad_x),max(0.,ny-pad_y);right,bottom=min(media['width'],nx+nw+pad_x),min(media['height'],ny+nh+pad_y)
    if right-left<2 or bottom-top<2:break
    if old and covered(old):
     ox,oy,ow,oh=old['box'];left=min(left,ox);top=min(top,oy);right=max(right,ox+ow);bottom=max(bottom,oy+oh)
    found=dict(obj,box=[left,top,right-left,bottom-top],polygons=None,visible=True,presence='visible',mask_allowed=True,detected=False,identity_ambiguous=False,privacy_coverage=True,appearance_padding=True,tracking_status='onset_safety_margin',timestamp=index/fps)
    found.pop('manual_contour',None)
    if old:frames[index].remove(old)
    frames[index].append(found);rows[index]=found;added+=1
 plan['appearance_padding_revision']=REVISION;plan['appearance_padding_frames']=added
 return added
