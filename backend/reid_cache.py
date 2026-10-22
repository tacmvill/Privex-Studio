"""Reuse ReID only for nearly identical, overlapping native crops over <=100 ms."""
import cv2,numpy as np
from videomask.persons import iou
class StableEncoder:
 def __init__(self,encoder,max_age=3):self.encoder=encoder;self.max_age=max_age;self.clock=0;self.frame_index=None;self.entries=[]
 def __call__(self,image,bboxes):
  self.clock=self.frame_index if self.frame_index is not None else self.clock+1;output=[None]*len(bboxes);missing=[];observations=[];used=set()
  for n,box in enumerate(bboxes):
   cx,cy,w,h=map(float,box[:4]);xywh=(cx-w/2,cy-h/2,w,h);x,y,bw,bh=map(round,xywh);crop=image[max(0,y):min(image.shape[0],y+bh),max(0,x):min(image.shape[1],x+bw)];appearance=cv2.resize(crop,(16,24)).astype(np.float32) if crop.size else None;observations.append((xywh,appearance));best=None
   if appearance is not None:
    for k,e in enumerate(self.entries):
     if k not in used and self.clock-e[0]<=self.max_age and iou(xywh,e[1])>.75 and float(np.abs(appearance-e[2]).mean())<=5:best=k;break
   if best is None:missing.append(n)
   else:output[n]=self.entries[best][3].copy();used.add(best);__import__('performance')._current and __import__('performance')._current.record('tracking/reid_cache_hit',0.)
  if missing:
   features=self.encoder(image,np.asarray([bboxes[n] for n in missing]))
   for n,feature in zip(missing,features):
    if feature is None:continue
    vector=np.asarray(feature,dtype=np.float32).copy();output[n]=vector
    if observations[n][1] is not None:self.entries.append((self.clock,*observations[n],vector.copy()))
  self.entries=[e for e in self.entries if self.clock-e[0]<=self.max_age][-256:];return output
