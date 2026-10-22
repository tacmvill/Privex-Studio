"""Close pinholes and preserve the small disconnected extremities of a body mask."""
import cv2,numpy as np

def protect(mask,box):
 result=np.uint8(mask>0)*255;x,y,w,h=map(round,box)
 if min(w,h)<=0:return result
 # Segmentation can miss the very bottom of a shoe: extend the observed lower
 # outline to the tracked foot line, without replacing the silhouette by a box.
 left,top=max(0,x),max(0,y);right,bottom=min(result.shape[1],x+w),min(result.shape[0],y+h)
 roi=result[top:bottom,left:right]
 if not roi.size:return result
 yy,xx=np.nonzero(roi)
 if len(yy):
  last=int(yy.max());lowest=xx[yy>=max(0,last-max(2,round(h*.08)))]
  if len(lowest) and h*.65<last<h*.98:
   pad=max(1,round(w*.04));a=max(0,int(lowest.min())-pad);b=min(roi.shape[1],int(lowest.max())+pad+1)
   roi[last:,a:b]=255
 radius=max(1,min(4,round(min(w,h)*.02)))
 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
 return cv2.dilate(cv2.morphologyEx(result,cv2.MORPH_CLOSE,kernel),kernel)


def guard_head(mask,box):
 # Body segmentation occasionally excludes a clearly visible head. Keep a
 # conservative head cap inside the tracked person envelope on every frame.
 x,y,w,h=box
 if min(w,h)>0:cv2.ellipse(mask,(round(x+w*.5),round(y+h*.16)),(max(1,round(w*.52)),max(1,round(h*.19))),0,0,360,255,-1)
 return mask
