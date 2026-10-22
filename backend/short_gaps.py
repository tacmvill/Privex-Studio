"""Fill <=100 ms holes only with measured forward/backward optical-flow evidence."""
from collections import defaultdict
import cv2,numpy as np
from identity_tracks import appearance
from videomask.persons import iou
from performance import timed
def repair(media,frames,cuts,job):
 from inspection import source_at
 tracks=defaultdict(list);repaired=0
 for index,row in enumerate(frames):
  for o in row:
   if o.get('detected',True) and o['id'].startswith('a'):tracks[o['id']].append((index,o))
 for ident,rows in tracks.items():
  scores=[o.get('confidence',o.get('detection_confidence',0)) for _,o in rows]
  if max(scores or [0])<.85 and (len(rows)<max(3,round(media['fps']*.35)) or float(np.median(scores))<.55):continue
  for (a,before),(b,after) in zip(rows,rows[1:]):
   if b-a<=1 or b-a>max(2,round(media['fps']*.1)+1) or any(a<c<=b for c in cuts) or before.get('identity_ambiguous') or after.get('identity_ambiguous'):continue
   if job.get('cancel'):raise InterruptedError('Suivi annulé.')
   with timed('tracking/short_gap_flow'):
    first=source_at(media,a);gray=cv2.cvtColor(first,cv2.COLOR_BGR2GRAY);x,y,w,h=map(round,before['box']);mask=np.zeros(gray.shape,np.uint8);mask[max(0,round(y+h*.15)):max(0,round(y+h*.85)),max(0,round(x+w*.15)):max(0,round(x+w*.85))]=255
    points=cv2.goodFeaturesToTrack(gray,maxCorners=35,qualityLevel=.03,minDistance=3,mask=mask)
    if points is None or len(points)<6:continue
    original=appearance(first,before['box']);box=np.array(before['box'],float);evidence=[];valid=True
    for index in range(a+1,b+1):
     image=source_at(media,index);current=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
     forward,status,_=cv2.calcOpticalFlowPyrLK(gray,current,points,None,winSize=(21,21),maxLevel=3)
     if forward is None:valid=False;break
     backward,back_status,_=cv2.calcOpticalFlowPyrLK(current,gray,forward,None,winSize=(21,21),maxLevel=3)
     if backward is None:valid=False;break
     keep=(status.reshape(-1)>0)&(back_status.reshape(-1)>0)&(np.linalg.norm(backward-points,axis=2).reshape(-1)<1.2)
     if keep.sum()<6 or keep.mean()<.75:valid=False;break
     matrix,inliers=cv2.estimateAffinePartial2D(points[keep],forward[keep],method=cv2.RANSAC,ransacReprojThreshold=2)
     if matrix is None or inliers is None or inliers.mean()<.75:valid=False;break
     corners=np.array([[[box[0],box[1]]],[[box[0]+box[2],box[1]+box[3]]]],np.float32);corners=cv2.transform(corners,matrix).reshape(2,2);updated=[float(corners[0,0]),float(corners[0,1]),float(corners[1,0]-corners[0,0]),float(corners[1,1]-corners[0,1])]
     ratio=updated[2]*updated[3]/max(1,box[2]*box[3]);feature=appearance(image,updated);similarity=float(feature@original) if feature is not None and original is not None else 0
     if not .75<ratio<1.3 or similarity<.88:valid=False;break
     box=np.asarray(updated);evidence.append((index,[round(v) for v in box],min(.95,float(inliers.mean())),similarity));points=forward[keep];gray=current
    if not valid or iou(tuple(box),tuple(after['box']))<.55:continue
    for index,box,confidence,similarity in evidence[:-1]:
     frames[index].append(dict(before,box=box,detected=False,visible=True,detection_confidence=0.,tracking_confidence=confidence,appearance_confidence=similarity,flow_verified=True));repaired+=1
 return repaired
