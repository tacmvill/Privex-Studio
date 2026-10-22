"""Bounded semantic + planar verification of printed faces, never immobility alone."""
import cv2,numpy as np
from collections import defaultdict

def inside(box,surface):
 x,y,w,h=box;X,Y,W,H=surface
 inter=max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y))
 return inter/max(1,w*h)>.95 and w*h/max(1,W*H)<.18

def supported(media,index,obj,surface,cuts,frame_at,diagnostic=None):
 box=obj['box'];body=obj.get('body_evidence',{}).get('box')
 if body:
  x,y,w,h=body;X,Y,W,H=surface;inter=max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y))
  if inter/max(1,w*h)<.8:return False
 fps=media['fps'];indices=[i for i in [index-round(fps),index+round(fps)] if 0<=i<media['frames'] and not any(min(i,index)<c<=max(i,index) for c in cuts)]
 if not indices:return False
 reference=frame_at(index);gray=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY);x,y,w,h=map(round,box);X,Y,W,H=map(round,surface)
 if x<0 or y<0 or x+w>gray.shape[1] or y+h>gray.shape[0]:return False
 patch=gray[y:y+h,x:x+w].astype(np.float32)
 if patch.size<64 or patch.std()<10:return False
 mask=np.full_like(gray,255);mask[y:y+h,x:x+w]=0
 points=cv2.goodFeaturesToTrack(gray,maxCorners=300,qualityLevel=.015,minDistance=5,mask=mask)
 if points is None or len(points)<12:return False
 for at in indices:
  target=cv2.cvtColor(frame_at(at),cv2.COLOR_BGR2GRAY)
  tracked,status,_=cv2.calcOpticalFlowPyrLK(gray,target,points,None,winSize=(31,31),maxLevel=4)
  if tracked is None:return False
  valid=status.reshape(-1)>0
  if valid.sum()<10:return False
  matrix,inliers=cv2.findHomography(points[valid],tracked[valid],cv2.RANSAC,2.)
  if diagnostic is not None:diagnostic.append({'frame':at,'inliers':float(inliers.mean()) if inliers is not None else 0})
  if matrix is None or inliers is None or inliers.mean()<.4 or int(inliers.sum())<15:return False
  aligned=cv2.warpPerspective(target,np.linalg.inv(matrix),(gray.shape[1],gray.shape[0]))[y:y+h,x:x+w].astype(np.float32)
  a=patch-patch.mean();b=aligned-aligned.mean();score=float((a*b).sum()/max(1,np.linalg.norm(a)*np.linalg.norm(b)))
  if diagnostic is not None:diagnostic[-1]['correlation']=score
  if score<.92:return False
 return True

def reject(media,frames,cuts,job,detector=None):
 from inspection import source_at
 from performance import phase,timed
 # Verify at most two frames for the whole analysis, sharing the loaded local model.
 tracks=defaultdict(list)
 for i,row in enumerate(frames):
  for o in row:
   if o.get('category',o['class']) in {'tête','visage'} and o.get('detected') and not o.get('privacy_coverage'):tracks[o['id']].append((i,o))
 static=[]
 for rows in tracks.values():
  if rows[-1][0]-rows[0][0]<media['fps']*.8 or len(rows)<media['fps']*.5:continue
  boxes=np.array([o['box'] for i,o in rows]);centers=boxes[:,:2]+boxes[:,2:]/2
  if np.max(np.ptp(centers,axis=0))>max(3,float(np.median(boxes[:,2:]))*.3):continue
  static.append(rows[len(rows)//2][0])
 if not static:return
 from advanced import installed,GroundedDetector
 if not installed().get('Grounding DINO'):job.setdefault('warnings',[]).append('Vérification des affiches indisponible : modèle local Grounding DINO absent.');return
 targets=sorted(set(static))[::max(1,len(static)//2)][:2]
 phase('Vérification des supports imprimés',len(targets),'images');model=None;rejected=set();zones=[];checked=0
 try:
  for index in targets:
   if job.get('cancel'):raise InterruptedError('Vérification annulée.')
   heads=[o for o in frames[index] if o.get('category',o['class']) in {'tête','visage'} and not o.get('privacy_coverage')]
   if not heads:continue
   cache={};frame_at=lambda at:cache.setdefault(at,source_at(media,at)) if at not in cache else cache[at]
   # Cheap texture/scene test precedes the heavy semantic model. A motionless
   # bounding box alone never justifies model loading or automatic rejection.
   heads=[o for o in heads if supported(media,index,o,[0,0,media['width'],media['height']],cuts,frame_at)]
   if not heads:continue
   if model is None:
    with timed('load/printed_supports'):model=GroundedDetector(['poster','advertisement','billboard'])
    __import__('performance').note_model('printed_supports','Grounding DINO local, PyTorch sur CPU ; deux images au maximum après le filtre de texture.', 'CPUExecutionProvider')
   with timed('inference/printed_supports'):surfaces=model.detect(source_at(media,index),.30)
   checked+=1
   cache={};frame_at=lambda at:cache.setdefault(at,source_at(media,at)) if at not in cache else cache[at]
   for obj in heads:
    if obj['id'] in rejected:continue
    surface=next((s['box'] for s in surfaces if inside(obj['box'],s['box'])),None)
    if surface and supported(media,index,obj,surface,cuts,frame_at):
     rejected.add(obj['id']);zones.append({'frame':index,'subject_id':obj['id'],'box':obj['box'],'surface':surface,'reason':'Visage imprimé : support publicitaire et texture solidaire du plan'})
 except InterruptedError:raise
 except Exception as exc:job.setdefault('warnings',[]).append('Vérification des affiches ignorée : '+str(exc)[:160])
 if rejected:
  for row in frames:row[:]=[o for o in row if o['id'] not in rejected]
 job['printed_faces']={'checked_frames':checked,'rejected_subjects':len(rejected),'evidence':zones}


