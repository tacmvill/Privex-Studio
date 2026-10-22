"""Conservative coverage of bounded holes in an already corroborated trajectory.

Coverage is a privacy envelope, not a claim that interpolated pixels are a face.
Never bridge a cut, an unresolved identity, a departure or a competing target.
"""
import math
import numpy as np
from videomask.persons import iou


def protect(media, observations, cuts, absent, bodies=None):
 fps=media['fps'];indices=sorted(i for i,o in observations.items() if o.get('visible') and not o.get('identity_ambiguous') and o.get('tracking_confidence',0)>=.6)
 filled=0
 for a,b in zip(indices,indices[1:]):
  gap=b-a-1
  if gap<1 or gap>round(fps*4) or any(a<c<=b for c in cuts) or any(i in absent for i in range(a+1,b)):continue
  left,right=observations[a],observations[b]
  if any(observations.get(i,{}).get('identity_ambiguous') for i in range(a,b+1)):continue
  # Longer occlusions require a unique re-identification, not just a reused ID.
  if gap>max(1,round(fps*.12)) and not (right.get('reidentified') and right.get('appearance_confidence',0)>=.9):continue
  old=np.asarray(left['box'],float);new=np.asarray(right['box'],float)
  distance=np.linalg.norm(new[:2]+new[2:]/2-old[:2]-old[2:]/2)
  diagonal=max(10,math.hypot(*old[2:]),math.hypot(*new[2:]))
  if distance>diagonal*(.8+min(1.5,gap/fps)):continue
  ratio=min(np.prod(old[2:])/max(1,np.prod(new[2:])),np.prod(new[2:])/max(1,np.prod(old[2:])))
  if ratio<.12:continue
  # The swept region covers a brief missed background head even between two
  # foreground masks. More prolonged intervals get a padded motion corridor.
  for i in range(a+1,b):
   t=(i-a)/(b-a);box=(1-t)*old+t*new
   if gap<=max(1,round(fps*.12)):
    x=min(old[0],new[0]);y=min(old[1],new[1]);w=max(old[0]+old[2],new[0]+new[2])-x;h=max(old[1]+old[3],new[1]+new[3])-y
   else:
    # Do not paint an invisible person onto a complete occluder. A longer
    # protective region also needs an observed, unique body at this instant.
    candidates=[]
    for body in (bodies or {}).get(i,[]):
     bx,by,bw,bh=body['box'];cx=box[0]+box[2]/2;cy=box[1]+box[3]/2
     if bx-bw*.2<=cx<=bx+bw*1.2 and by-bh*.2<=cy<=by+bh*.8:candidates.append(body)
    if len(candidates)!=1:continue
    if (left.get('category') or left.get('class','')).casefold() in {'tête','visage'}:
     # Body context confirms identity/visibility; its torso is not head geometry.
     x,y,w,h=box
    else:x,y,w,h=candidates[0]['box']
   pad=.12+min(.3,gap/fps*.08);x-=w*pad;y-=h*pad;w*=1+2*pad;h*=1+2*pad
   x=max(0,x);y=max(0,y);w=min(media['width']-x,w);h=min(media['height']-y,h)
   observations[i]=dict(left,box=[round(x),round(y),round(w),round(h)],detected=False,visible=True,tracking_confidence=.65,identity_ambiguous=False,privacy_coverage=True,coverage_evidence='bracketed-short-gap' if gap<=max(1,round(fps*.12)) else 'unique-reidentification',detection_confidence=0.,polygons=[])
   filled+=1
 return filled
