"""Batched Lucas–Kanade foreground evidence between detector observations.

One forward/backward optical-flow pair for all subjects, not a CSRT per subject.
Only measured, appearance-validated motion is returned; no blind Kalman boxes.
"""
import cv2,numpy as np

def feature(image,box):
 x,y,w,h=map(round,box);crop=image[max(0,y):min(image.shape[0],y+h),max(0,x):min(image.shape[1],x+w)]
 if not crop.size:return None
 tile=cv2.resize(crop,(24,24));hsv=cv2.cvtColor(tile,cv2.COLOR_BGR2HSV);hist=cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256]).ravel();hist/=max(1e-6,np.linalg.norm(hist))
 gray=cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY).astype(float).ravel();gray-=gray.mean();gray/=max(1e-6,np.linalg.norm(gray))
 return hist,gray

class SparseMotion:
 def __init__(self,fps):self.fps=fps;self.previous=None;self.tracks={};self.scale=1.;self.current=None
 def image(self,frame):
  self.scale=min(1.,960/max(frame.shape[:2]));return cv2.resize(frame,None,fx=self.scale,fy=self.scale) if self.scale<1 else frame
 @staticmethod
 def points(gray,box):
  x,y,w,h=map(round,box);margin=.08;x1=max(0,round(x+w*margin));y1=max(0,round(y+h*margin));x2=min(gray.shape[1],round(x+w*(1-margin)));y2=min(gray.shape[0],round(y+h*(1-margin)))
  if x2-x1<5 or y2-y1<5:return None
  points=cv2.goodFeaturesToTrack(gray[y1:y2,x1:x2],maxCorners=20,qualityLevel=.02,minDistance=2)
  return points+np.float32([x1,y1]) if points is not None else None
 def infer(self,frame):
  image=self.image(frame);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);self.current=(image,gray);valid=[(ident,state) for ident,state in self.tracks.items() if state['points'] is not None and len(state['points'])>=4 and state['missing']<round(self.fps*3)]
  if self.previous is None or not valid:return []
  points=np.concatenate([s['points'] for _,s in valid]);forward,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,points,None,winSize=(25,25),maxLevel=3)
  if forward is None:return []
  back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,forward,None,winSize=(25,25),maxLevel=3)
  if back is None:return []
  good=(status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1.2);offset=0;result=[]
  for ident,state in valid:
   n=len(state['points']);keep=good[offset:offset+n];old=points[offset:offset+n];new=forward[offset:offset+n];offset+=n
   if keep.sum()<4 or keep.mean()<.7:continue
   affine,inliers=cv2.estimateAffinePartial2D(old[keep],new[keep],method=cv2.RANSAC,ransacReprojThreshold=2)
   if affine is None or inliers is None or inliers.mean()<.75:continue
   scale=float(np.linalg.norm(affine[:,0]))
   if not .8<scale<1.25:continue
   x,y,w,h=np.asarray(state['obj']['box'])*self.scale;corners=np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]],np.float32);corners=cv2.transform(corners,affine).reshape(-1,2);left,top=corners.min(axis=0);right,bottom=corners.max(axis=0)
   box=[float(left),float(top),float(right-left),float(bottom-top)]
   if left<0 or top<0 or right>gray.shape[1] or bottom>gray.shape[0] or min(box[2:])<4:continue
   current=feature(image,box);reference=state['feature']
   if current is None or reference is None:continue
   color=float(current[0]@reference[0]);texture=max(0.,float(current[1]@reference[1]))
   if color<.8 or .6*color+.4*texture<.65:continue
   native=[round(v/self.scale) for v in box];obj=dict(state['obj'],id=ident,box=native,detected=False,visible=True,detection_confidence=0.,tracking_confidence=min(.98,.55+.43*float(inliers.mean())*color),appearance_confidence=color,flow_verified=True,observed_motion=True)
   if obj.get('polygons'):
    ox,oy,ow,oh=state['obj']['box'];nx,ny,nw,nh=native;obj['polygons']=[[[nx+(px-ox)*nw/max(1,ow),ny+(py-oy)*nh/max(1,oh)] for px,py in poly] for poly in obj['polygons']]
   result.append(obj)
  return result
 def commit(self,rows,frame):
  image,gray=self.current if self.current is not None else (None,None)
  if image is None:image=self.image(frame);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  following={}
  for obj in rows:
   if obj.get('identity_ambiguous') or not obj.get('visible',True):continue
   ident=obj['id'];prior=self.tracks.get(ident);observed=obj.get('detected',True) or obj.get('body_verified')
   if prior is None and (obj.get('confidence',0)<.3 or obj.get('continuation_only')):continue
   box=np.asarray(obj['box'])*self.scale;reference=feature(image,box) if observed and obj.get('confidence',1)>=.3 or prior is None else prior['feature']
   following[ident]={'obj':dict(obj),'points':self.points(gray,box),'feature':reference,'missing':0 if observed else prior['missing']+1 if prior else 1}
  self.tracks=following;self.previous=gray;self.current=None
