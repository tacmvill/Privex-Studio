"""Fuse face/head detections with person context; never emit body targets for head-only requests."""
from videomask.core import _clip_box
from videomask.persons import iou

def same_head(head,face):
 x,y,w,h=head;X,Y,W,H=face
 # A huge low-confidence head box must not swallow several nearby faces.
 return (x<=X+W/2<=x+w and y<=Y+H/2<=y+h and .15<=W*H/max(1,w*h)<=2.5) or iou(tuple(head),tuple(face))>.4

def belongs(head,body):
 x,y,w,h=head;X,Y,W,H=body
 return X-W*.08<=x+w/2<=X+W*1.08 and Y-H*.04<=y+h/2<=Y+H*.42 and w<=W*1.3 and h<=H*.65

def refine(frame,rows,face,cfg,regional=True,allow_body_estimates=True):
 h,w=frame.shape[:2];heads=[o for o in rows if o['class']=='tête'];bodies=[o for o in rows if o['class']=='personne' and o['confidence']>=max(.5,cfg.confidence)]
 # Keep weak head proposals only when supported by a person. SCRFD face proposals
 # remain independent: a close face can legitimately have no visible body.
 # Fuse a face and the corresponding whole head before adding body estimates.
 fused=[]
 for o in sorted(heads,key=lambda o:(not o.get('geometry_estimated',False),o['confidence']),reverse=True):
  prior=next((p for p in fused if same_head(p['box'],o['box'])),None)
  if prior is None:fused.append(o)
  else:prior['confidence']=max(prior['confidence'],o['confidence'])
 heads=[o for o in fused if o.get('geometry_estimated') or o['confidence']>=.55 or any(belongs(o['box'],b['box']) for b in bodies)]
 missing=[b for b in bodies if not any(belongs(o['box'],b['box']) for o in heads)]
 # Group native upper-body crops so a crowded area does not trigger one model per person.
 crops=[]
 for body in sorted(missing,key=lambda b:b['confidence'],reverse=True):
  x,y,bw,bh=body['box'];box=_clip_box((round(x-bw*.15),round(y-bh*.08),round(bw*1.3),round(bh*.52)),w,h)
  if not box:continue
  if any(iou(tuple(box),tuple(c))>.35 for c in crops):continue
  crops.append(box)
 # Complementary detections are bounded per detector frame, not every output frame.
 if face and regional:
  for x,y,bw,bh in crops[:(1 if cfg.profile=='rapide' else 3)]:
   crop=frame[y:y+bh,x:x+bw]
   found=face.scrfd.detect(crop,size=640,threshold=.40) if face.scrfd else face.yunet.detect(crop)
   for (X,Y,W,H),score in found:
    box=_clip_box((round(x+X-W*.15),round(y+Y-H*.3),round(W*1.3),round(H*1.4)),w,h)
    if not box or any(same_head(o['box'],box) for o in heads):continue
    heads.append({'box':list(box),'class':'tête','class_id':102,'confidence':score,'geometry_estimated':True,'model_source':'SCRFD · recherche régionale' if face.scrfd else 'YuNet · recherche régionale','method':'Visage dans la région de la personne'})
 for body in bodies if allow_body_estimates else []:
  if any(belongs(o['box'],body['box']) for o in heads):continue
  # Conservative upper-body estimate also covers rear/profile heads missed by face models.
  from videomask.heads import person_to_head
  box=person_to_head(tuple(body['box']),w,h)
  heads.append({'box':list(box),'class':'tête','class_id':102,'confidence':body['confidence']*.85,'geometry_estimated':True,'model_source':'Personne · tête estimée','method':'Région protectrice estimée depuis la personne','body_evidence':{'box':body['box'],'confidence':body['confidence'],'model_source':body.get('model_source','YOLO')}})
 return [o for o in rows if o['class']!='tête']+heads
