"""Foreground-seeded tracking for user rectangles; no background-template learning."""
import cv2
import numpy as np
from videomask.custom_tracking import ObjectFollower


def foreground(frame,box):
 """GrabCut in a bounded contextual crop. A failed separation stays explicit."""
 x,y,w,h=map(round,box);height,width=frame.shape[:2];pad=max(8,round(max(w,h)*.25))
 x0=max(0,x-pad);y0=max(0,y-pad);x1=min(width,x+w+pad);y1=min(height,y+h+pad)
 crop=frame[y0:y1,x0:x1];mask=np.zeros(frame.shape[:2],np.uint8)
 if w<6 or h<6 or not crop.size:return mask,False
 scale=min(1,320/max(crop.shape[:2]));small=cv2.resize(crop,None,fx=scale,fy=scale)
 rect=(max(1,round((x-x0)*scale)),max(1,round((y-y0)*scale)),max(2,round(w*scale)),max(2,round(h*scale)))
 rx,ry,rw,rh=rect;rect=(rx,ry,min(rw,small.shape[1]-rx-1),min(rh,small.shape[0]-ry-1))
 if min(rect[2:])<2:return mask,False
 labels=np.zeros(small.shape[:2],np.uint8)
 try:cv2.grabCut(small,labels,rect,None,None,3,cv2.GC_INIT_WITH_RECT)
 except cv2.error:return mask,False
 fg=np.uint8((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD))*255
 count,components,stats,centres=cv2.connectedComponentsWithStats(fg)
 if count<=1:return mask,False
 # Choose a substantial central component; never the surrounding background.
 cx=(x+w/2-x0)*scale;cy=(y+h/2-y0)*scale
 chosen=max(range(1,count),key=lambda i:stats[i,cv2.CC_STAT_AREA]/(1+np.linalg.norm(centres[i]-[cx,cy])/max(1,w*scale)))
 selected=np.uint8(components==chosen)*255
 native=cv2.resize(selected,(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST);mask[y0:y1,x0:x1]=native
 # Restrict foreground to the user's rectangle, even if GrabCut bleeds outward.
 allowed=np.zeros_like(mask);allowed[max(0,y):min(height,y+h),max(0,x):min(width,x+w)]=255;mask=cv2.bitwise_and(mask,allowed)
 ratio=np.count_nonzero(mask)/max(1,w*h)
 return mask,bool(.04<=ratio<=.96)


def polygon(mask):
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 if not contours:return []
 result=[]
 for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:128]:
  if cv2.contourArea(contour)<1:continue
  contour=cv2.approxPolyDP(contour,max(.5,cv2.arcLength(contour,True)*.003),True)
  if len(contour)>=3:result.append(contour.reshape(-1,2).astype(float).tolist())
 return result


class ForegroundFollower:
 """Object points drive motion; CSRT is a validated fallback, not scenery memory."""
 def __init__(self,frame,box):
  self.box=tuple(map(round,box));self.gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);self.mask,self.separated=foreground(frame,self.box)
  if not self.separated:
   x,y,w,h=self.box;self.mask[:]=0;self.mask[max(0,y):y+h,max(0,x):x+w]=255
  self.outline=polygon(self.mask);self.polygons=self.outline;self.points=self._points(self.gray,self.mask)
  self.fallback=ObjectFollower(frame,self.box);self.misses=0;self.confidence=1.;self.age=0
  self.template=frame[self.box[1]:self.box[1]+self.box[3],self.box[0]:self.box[0]+self.box[2]].copy()
  roi=self.mask[self.box[1]:self.box[1]+self.box[3],self.box[0]:self.box[0]+self.box[2]];self.template_mask=roi.copy()
  self.feature=self._feature(frame,self.mask)
  # OpenCV CSRT supports an explicit initial segmentation mask.
  tracker=self.fallback.csrt
  if self.separated and tracker is not None and hasattr(tracker,'setInitialMask'):
   try:
    from videomask.custom_tracking import _create_csrt
    tracker=_create_csrt();tracker.setInitialMask(roi);tracker.init(frame,self.box);self.fallback.csrt=tracker
   except cv2.error:pass

 @staticmethod
 def _points(gray,mask):return cv2.goodFeaturesToTrack(gray,maxCorners=80,qualityLevel=.015,minDistance=3,mask=cv2.dilate(mask,np.ones((3,3),np.uint8)))
 @staticmethod
 def _feature(frame,mask):
  hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);hist=cv2.calcHist([hsv],[0,1],mask,[16,8],[0,180,0,256]).reshape(-1)
  return hist/max(1e-6,np.linalg.norm(hist))

 def update(self,frame):
  self.age+=1;gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);matrix=None;quality=0.
  if self.points is not None and len(self.points)>=4 and not self.misses:
   forward,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,winSize=(25,25),maxLevel=3)
   if forward is not None:
    back,valid,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,forward,None,winSize=(25,25),maxLevel=3)
    if back is not None:
     keep=(ok.ravel()>0)&(valid.ravel()>0)&(np.linalg.norm(back-self.points,axis=2).ravel()<1.2)
     if keep.sum()>=3 and keep.mean()>=.7:
      candidate,inliers=cv2.estimateAffinePartial2D(self.points[keep],forward[keep],method=cv2.RANSAC,ransacReprojThreshold=2)
      if candidate is not None and inliers is not None and inliers.mean()>=.7:
       scale=float(np.linalg.norm(candidate[:,0]));moved=cv2.warpAffine(self.mask,candidate,(gray.shape[1],gray.shape[0]),flags=cv2.INTER_NEAREST)
       similarity=float(self.feature@self._feature(frame,moved)) if np.any(moved) else 0.
       if .8<scale<1.25 and similarity>=.78:matrix=candidate;quality=min(.98,float(inliers.mean())*similarity)
  if matrix is not None:
   x,y,w,h=self.box;corners=np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]],np.float32);corners=cv2.transform(corners,matrix).reshape(-1,2)
   left,top=corners.min(axis=0);right,bottom=corners.max(axis=0)
   self.mask=cv2.warpAffine(self.mask,matrix,(gray.shape[1],gray.shape[0]),flags=cv2.INTER_NEAREST)
   self.box=(round(left),round(top),round(right-left),round(bottom-top));self.polygons=polygon(self.mask);self.points=self._points(gray,self.mask);self.gray=gray;self.misses=0;self.confidence=max(.65,quality)
   # Only refresh CSRT on measured foreground movement; its template cannot adapt to scenery.
   if self.age%10==0:self.fallback=ObjectFollower(frame,self.box)
   return self.box
  proposed=self.fallback.update(frame) if not self.misses else self._recover(frame)
  if proposed is not None:
   x,y,w,h=map(round,proposed);mask=np.zeros_like(gray)
   if w>0 and h>0:
    local=cv2.resize(self.template_mask,(w,h),interpolation=cv2.INTER_NEAREST);xa=max(0,x);ya=max(0,y);xb=min(gray.shape[1],x+w);yb=min(gray.shape[0],y+h)
    if xb>xa and yb>ya:mask[ya:yb,xa:xb]=local[ya-y:yb-y,xa-x:xb-x]
   similarity=float(self.feature@self._feature(frame,mask)) if np.any(mask) else 0.
   if similarity>=.85 and (self.misses or self.fallback.confidence>=.6):
    self.box=tuple(map(round,proposed));self.mask=mask;self.polygons=polygon(mask);self.gray=gray;self.points=self._points(gray,mask);self.misses=0;self.confidence=min(.95,similarity);return self.box
  self.misses+=1;self.confidence=0.;return None

 def _recover(self,frame):
  # Bounded, ambiguity-tested search around the last subject, not the whole scene.
  if self.misses%5!=1 or not self.template.size or self.template.std()<8:return None
  x,y,w,h=self.box;pad=max(w,h)*min(3,1+self.misses*.1);xa=max(0,round(x-pad));ya=max(0,round(y-pad));xb=min(frame.shape[1],round(x+w+pad));yb=min(frame.shape[0],round(y+h+pad));search=frame[ya:yb,xa:xb]
  if not search.size or min(w,h)<5:return None
  scale=min(1,640/max(search.shape[:2]));small=cv2.resize(search,None,fx=scale,fy=scale);template=cv2.resize(self.template,(max(5,round(w*scale)),max(5,round(h*scale))));mask=cv2.resize(self.template_mask,(template.shape[1],template.shape[0]),interpolation=cv2.INTER_NEAREST)
  if template.shape[0]>small.shape[0] or template.shape[1]>small.shape[1]:return None
  scores=cv2.matchTemplate(small,template,cv2.TM_CCORR_NORMED,mask=mask);scores=np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1);_,best,_,pos=cv2.minMaxLoc(scores)
  second=scores.copy();px,py=pos;second[max(0,py-template.shape[0]//2):py+template.shape[0]//2+1,max(0,px-template.shape[1]//2):px+template.shape[1]//2+1]=-1
  if best<.94 or best-float(second.max())<.05:return None
  return (xa+round(px/scale),ya+round(py/scale),w,h)
