"""Local detectors + ByteTrack; V9 face and manual tracking are reused."""
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from videomask.core import YoloDetector,_clip_box
from videomask.faces import HighRecallFaceDetector
from videomask.custom_tracking import CustomObject,plan_custom_objects

ROOT=Path(__file__).resolve().parent
COCO_NAMES=['personne','vélo','voiture','moto','avion','bus','train','camion','bateau','feu de circulation','borne incendie','panneau stop','parcmètre','banc','oiseau','chat','chien','cheval','mouton','vache','éléphant','ours','zèbre','girafe','sac à dos','parapluie','sac à main','cravate','valise','frisbee','skis','snowboard','ballon','cerf-volant','batte de baseball','gant de baseball','skateboard','planche de surf','raquette de tennis','bouteille','verre à vin','tasse','fourchette','couteau','cuillère','bol','banane','pomme','sandwich','orange','brocoli','carotte','hot-dog','pizza','donut','gâteau','chaise','canapé','plante en pot','lit','table','toilettes','télévision','ordinateur portable','souris','télécommande','clavier','téléphone','micro-ondes','four','grille-pain','évier','réfrigérateur','livre','horloge','vase','ciseaux','ours en peluche','sèche-cheveux','brosse à dents']
CLASS_IDS={name:i for i,name in enumerate(COCO_NAMES)}
PROFILES={'rapide':{'quality':'standard'},'moyen':{'quality':'precise'},'extreme':{'quality':'extreme'}}
PALETTE=['#ff453a','#0a84ff','#ffd60a','#30d158','#ff9f0a','#bf5af2','#64d2ff','#ff375f']

class LegacyDetector:
 def __init__(self,cfg):
  from vision_models import ModelManager
  self.manager=ModelManager(cfg.profile)
  self.cfg=cfg;self.face=None;self.yolo=None;self.plate=None;self.head=None;self.extra=None;self.grounded=None
  if cfg.grounding_prompts and cfg.scope=='tous':
   from advanced import GroundedDetector
   self.grounded=GroundedDetector(cfg.grounding_prompts)
  if cfg.heads and cfg.scope=='tous':
   from videomask.heads import YoloHeadONNX
   try:self.head=YoloHeadONNX(ROOT/'models/head_detector.onnx');self.manager.models.append('HollywoodHeads')
   except Exception:self.manager.warnings.append('Détecteur de têtes indisponible : estimation depuis les visages ; les têtes de dos ne sont pas garanties.')
  if (cfg.extended or cfg.ocr or cfg.brands) and cfg.scope=='tous':
   from extended import ExtendedDetector
   self.extra=ExtendedDetector(cfg.extended,cfg.ocr,cfg.brands)
  if (cfg.faces or cfg.heads) and cfg.scope=='tous':
   try:self.face=HighRecallFaceDetector(require_scrfd=True)
   except Exception:
    self.face=HighRecallFaceDetector(require_scrfd=False);self.manager.warnings.append('SCRFD indisponible : détection des visages par YuNet.')
   self.manager.models.append('YuNet/SCRFD' if self.face.scrfd else 'YuNet')
  if (cfg.classes or cfg.evidence_classes or cfg.heads) and cfg.scope=='tous':self.yolo=self.manager.load_objects()
  if cfg.plates and cfg.scope=='tous':
   import onnxruntime as ort
   self.plate=ort.InferenceSession(str(ROOT/'models/yolox_plates_s.onnx'),providers=['CPUExecutionProvider'])
  self.provider=self.yolo.provider if self.yolo else (self.face.provider if self.face else self.head.provider if self.head else 'OpenCV CPU / CSRT')
  self._face_pool=None
  if cfg.profile=='rapide' and self.face and self.face.scrfd is None and self.head:
   from concurrent.futures import ThreadPoolExecutor
   import weakref
   self._face_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='privex-face-cpu')
   weakref.finalize(self,self._face_pool.shutdown,wait=False,cancel_futures=True)
 def detect(self,frame,face_quality=None,human_only=False,skip_body=False):
  result=[];h,w=frame.shape[:2]
  face_future=self._face_pool.submit(self.face.detect_scored,frame,face_quality or PROFILES[self.cfg.profile]['quality']) if self._face_pool else None
  if self.grounded and not human_only:result.extend({**o,'model_source':'Grounding DINO'} for o in self.grounded.detect(frame,self.cfg.confidence))
  if self.head:
   for box,score in self.head.detect_scored(frame,threshold=.30):result.append({'box':list(box),'class':'tête','class_id':102,'confidence':score,'model_source':'HollywoodHeads'})
  if self.extra and not human_only:result.extend(self.extra.detect(frame,self.cfg.confidence))
  if self.face:
   for (x,y,bw,bh),face_confidence in (face_future.result() if face_future else self.face.detect_scored(frame,face_quality or PROFILES[self.cfg.profile]['quality'])):
    if self.cfg.heads:
     box=_clip_box((round(x-bw*.15),round(y-bh*.3),round(bw*1.3),round(bh*1.4)),w,h)
     if box:result.append({'box':list(box),'class':'tête','class_id':102,'confidence':face_confidence,'geometry_estimated':True,'review_required':face_confidence<.85,'model_source':'YuNet/SCRFD · tête estimée','method':'Tête estimée depuis le visage'})
    else:result.append({'box':[x,y,bw,bh],'class':'visage','class_id':100,'confidence':face_confidence,'model_source':'YuNet/SCRFD'})
  if self.yolo and not skip_body:
   wanted={0} if human_only else {CLASS_IDS[c] for c in [*self.cfg.classes,*self.cfg.evidence_classes] if c in CLASS_IDS}
   if not human_only and set(self.cfg.classes)&{'voiture','bus','camion'}:wanted|={2,5,7}  # Reuse the same YOLO output for an existing vehicle's class changes.
   if self.cfg.heads:wanted.add(0)
   self.yolo.detect(frame,wanted,.10)
   pred=self.yolo._cached_prediction
   if pred.shape[0]==84:pred=pred.T
   scale=min(640/w,640/h);left=(640-round(w*scale))//2;top=(640-round(h*scale))//2
   ids=np.argmax(pred[:,4:],axis=1);scores=pred[np.arange(len(pred)),ids+4]
   for cls in wanted:
    proposals=[];conf=[]
    for i in np.flatnonzero((ids==cls)&(scores>=.10)):
     cx,cy,bw,bh=pred[i,:4];box=_clip_box((round((cx-bw/2-left)/scale),round((cy-bh/2-top)/scale),round(bw/scale),round(bh/scale)),w,h)
     if box:proposals.append(list(box));conf.append(float(scores[i]))
    keep=cv2.dnn.NMSBoxes(proposals,conf,.10,.45) if proposals else []
    name=next(k for k,v in CLASS_IDS.items() if v==cls)
    for i in np.asarray(keep).reshape(-1):result.append({'box':proposals[int(i)],'class':name,'class_id':cls,'confidence':conf[int(i)]})
  if self.plate and not human_only:
   scale=min(640/w,640/h);canvas=np.full((640,640,3),114,np.uint8);nw,nh=round(w*scale),round(h*scale);canvas[:nh,:nw]=cv2.resize(frame,(nw,nh))
   tensor=np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32)
   pred=np.squeeze(self.plate.run(None,{self.plate.get_inputs()[0].name:tensor})[0])
   # Pinned Autolane ONNX export ends in raw heads (no grid/Exp decoding in its graph).
   if pred.shape[1]!=6:raise ValueError('Sortie du modèle plaques non reconnue.')
   grids=[];strides=[]
   for stride in [8,16,32]:
    xx,yy=np.meshgrid(np.arange(640//stride),np.arange(640//stride));grids.append(np.stack((xx,yy),-1).reshape(-1,2));strides.append(np.full((xx.size,1),stride))
   grid=np.concatenate(grids);stride=np.concatenate(strides);pred[:,:2]=(pred[:,:2]+grid)*stride;pred[:,2:4]=np.exp(np.clip(pred[:,2:4],-20,20))*stride
   proposals=[];conf=[]
   for row in pred:
    score=float(row[4]*row[5])
    if score<self.cfg.confidence:continue
    cx,cy,bw,bh=row[:4];box=_clip_box((round((cx-bw/2)/scale),round((cy-bh/2)/scale),round(bw/scale),round(bh/scale)),w,h)
    if box:proposals.append(list(box));conf.append(score)
   keep=cv2.dnn.NMSBoxes(proposals,conf,self.cfg.confidence,.45) if proposals else []
   for i in np.asarray(keep).reshape(-1):result.append({'box':proposals[int(i)],'class':'plaque','class_id':101,'confidence':conf[int(i)]})
  if self.cfg.heads:result=__import__('head_fusion').refine(frame,result,self.face,self.cfg,regional=face_quality not in {'regional','precise'},allow_body_estimates=not human_only)
  self.provider=self.yolo.provider if self.yolo else (self.face.scrfd.provider if self.face and self.face.scrfd else 'OpenCV CPU (YuNet)' if self.face else self.head.provider if self.head else 'OpenCV CPU / CSRT')
  from videomask.persons import iou
  fused=[]
  for obj in result:
   category=obj.get('category',obj['class']).casefold();prior=next((o for o in fused if o.get('category',o['class']).casefold()==category and iou(tuple(o['box']),tuple(obj['box']))>.65),None)
   if prior is None:fused.append(obj)
   elif obj['confidence']>prior['confidence']:prior.update(obj)
  return fused

class MultiTracker:
 def __init__(self,fps,namespace="",algorithm='bytetrack'):
  self.namespace=namespace
  from trackers import ByteTrackTracker,OCSORTTracker
  self.algorithm=algorithm;self.ctor=OCSORTTracker if algorithm=='ocsort' else ByteTrackTracker;self.groups={};self.fps=fps;self.pending=[];self.provisional=0;self.labels={};self.colors={}
 def update(self,objects,frame=None):
  import supervision as sv
  from videomask.persons import iou
  labels={d['class_id']:d['class'] for d in objects};self.labels.update(labels);result=[]
  for cls in set(labels)|set(self.groups):
   group=[d for d in objects if d['class_id']==cls]
   if cls not in self.groups:
    common={'frame_rate':self.fps,'lost_track_buffer':240,'minimum_consecutive_frames':1,'high_conf_det_threshold':.1 if self.algorithm=='ocsort' else .3}
    self.groups[cls]=self.ctor(**common,**({'minimum_iou_threshold':.15,'direction_consistency_weight':.2} if self.algorithm=='ocsort' else {'track_activation_threshold':.3}))
   if group:
    xyxy=np.array([[b['box'][0],b['box'][1],b['box'][0]+b['box'][2],b['box'][1]+b['box'][3]] for b in group],dtype=np.float32)
    detections=sv.Detections(xyxy=xyxy,confidence=np.array([b['confidence'] for b in group]),class_id=np.full(len(group),cls))
   else:detections=sv.Detections.empty()
   tracked=self.groups[cls].update(detections)
   for i,box in enumerate(tracked.xyxy):
    raw_id=int(tracked.tracker_id[i]);tid=raw_id+cls*10000
    coords=[round(box[0]),round(box[1]),round(box[2]-box[0]),round(box[3]-box[1])]
    if raw_id<0:
     self.provisional+=1;ident=f'a{self.namespace}-p{cls}-{self.provisional}';color=PALETTE[self.provisional%len(PALETTE)]
    else:
     ident=f'a{self.namespace}{tid}'
     if ident not in self.colors:self.colors[ident]=PALETTE[len(self.colors)%len(PALETTE)]
     color=self.colors[ident]
     # ByteTrack confirms new tracks on the next update. Backfill the prior frame's provisional ID.
     from videomask.persons import iou
     candidates=[p for p in self.pending if p['class_id']==cls and '-p' in p['id']]
     if candidates:
      prior=max(candidates,key=lambda p:iou(tuple(p['box']),tuple(coords)))
      if iou(tuple(prior['box']),tuple(coords))>.1:prior.update(id=ident,color=color)
    nearby=max(group,key=lambda o:iou(tuple(o['box']),tuple(coords))) if group else {}
    extra={k:nearby[k] for k in ['category','method','text','review_required','confidence_estimated','model_source','model_confidences','auxiliary','body_evidence','existence_verified','verification_uncertain','continuation_only','vehicle_target','model_category','vehicle_partial','privacy_coverage'] if k in nearby}
    result.append({'id':ident,'box':coords,**extra,'class_id':cls,'class':nearby.get('class',self.labels.get(cls,'objet')),'confidence':float(tracked.confidence[i]),'detected':bool(nearby and iou(tuple(nearby['box']),tuple(coords))>.3),'color':color})
  self.pending=result
  return result

_segment=None
def mask_frame(source,objects,cfg):
 global _segment
 from types import SimpleNamespace
 from videomask.persons import iou,PersonSegmentationONNX
 from videomask.contours import grabcut_selected,tiled_silhouette
 out=source.copy();h,w=out.shape[:2];instances=None
 for obj in objects:
  settings=cfg.model_dump();settings.update(obj.get('style') or {});style=SimpleNamespace(**settings)
  if style.effect=='pixels_noirs':style.shape='pixelise'
  if obj.get('manual_contour') and style.effect!='pixels_noirs':style.shape='lisse'
  if obj.get('privacy_coverage'):
   if not obj.get('polygons'):style.shape='rectangle'
   style.feather=0;style.size=max(100,style.size)
  if obj.get('category',obj.get('class'))=='personne':style.size=max(100,style.size)
  bx,by,bw,bh=obj['box'];factor=style.size/100
  box=(round(bx-bw*(factor-1)/2),round(by-bh*(factor-1)/2),round(bw*factor),round(bh*factor))
  clipped=_clip_box(box,w,h,style.margin/100)
  if not clipped:continue
  x,y,bw,bh=clipped;roi=out[y:y+bh,x:x+bw]
  if style.effect in {'noir','pixels_noirs'}:masked=np.zeros_like(roi)
  elif style.effect in {'pixels','pixels_gris','pixels_rgb'}:
   block=max(2,int((2+(style.tile_size*1.26 if style.tile_size is not None else style.intensity*.32))/2+.5)*2);small=cv2.resize(roi,(max(1,bw//block),max(1,bh//block)),interpolation=cv2.INTER_AREA)
   if style.effect in {'pixels_gris','pixels_rgb'}:
    light=np.rint(small[:,:,2]*.299+small[:,:,1]*.587+small[:,:,0]*.114)
    if style.effect=='pixels_gris':small=np.repeat(np.where(light<85,0,np.where(light<170,128,255))[:,:,None],3,axis=2).astype(np.uint8)
    else:small=np.rint(light[:,:,None]*np.array(rgb_color(style.mosaic_color)[::-1])[None,None,:]/255).astype(np.uint8)
   masked=cv2.resize(small,(bw,bh),interpolation=cv2.INTER_NEAREST)
  else:
   scale=min(1.,320/max(bw,bh));small=cv2.resize(roi,(max(1,round(bw*scale)),max(1,round(bh*scale))),interpolation=cv2.INTER_AREA)
   if style.effect=='gaussien':blurred=cv2.GaussianBlur(small,(0,0),sigmaX=max(.5,style.intensity*.6*scale))
   else:
    kernel=max(3,round(style.intensity*.9*scale));blurred=cv2.blur(small,(kernel,kernel))
   masked=cv2.resize(blurred,(bw,bh),interpolation=cv2.INTER_LINEAR)
  mask=np.zeros((h,w),np.uint8);mask[y:y+bh,x:x+bw]=255
  if style.shape!='rectangle':
   chosen=None
   if obj.get('polygons'):
    chosen=np.zeros((h,w),np.uint8);cv2.fillPoly(chosen,[np.array(p,np.int32) for p in obj['polygons']],255)
   class_id=CLASS_IDS.get(obj.get('category',obj['class']),obj.get('class_id',-1))
   if chosen is None and 0<=class_id<80 and not obj.get('contour_attempted'):
    if _segment is None:_segment=PersonSegmentationONNX()
    if instances is None:instances=_segment.predict(source,classes={CLASS_IDS.get(o.get('category',o['class']),o.get('class_id',-1)) for o in objects if 0<=CLASS_IDS.get(o.get('category',o['class']),o.get('class_id',-1))<80})
    matching=[p for p in instances if p.class_id==class_id and iou(p.box,tuple(obj['box']))>.25]
    if matching:
     chosen=max(matching,key=lambda p:iou(p.box,tuple(obj['box']))).mask
     if obj.get('category',obj['class'])=='personne':chosen=__import__('body_contour').protect(chosen,obj['box'])
   if chosen is None and not obj.get('contour_attempted'):
    # V9 GrabCut fallback, bounded to 480px for predictable latency.
    scale=min(1.,480/max(h,w));small=cv2.resize(source,(round(w*scale),round(h*scale)))
    chosen=cv2.resize(grabcut_selected(small,tuple(round(v*scale) for v in obj['box']),0),(w,h),interpolation=cv2.INTER_NEAREST)
   if chosen is None:chosen=np.zeros((h,w),np.uint8)
   if not np.any(chosen):chosen[y:y+bh,x:x+bw]=255
   if obj.get('category',obj['class'])=='personne':chosen=__import__('body_contour').guard_head(chosen,obj['box'])
   if factor!=1:
    cx=bx+obj['box'][2]/2;cy=by+obj['box'][3]/2
    chosen=cv2.warpAffine(chosen,np.array([[factor,0,cx*(1-factor)],[0,factor,cy*(1-factor)]],np.float32),(w,h),flags=cv2.INTER_NEAREST)
   mask=chosen
   dilation=round(max(obj['box'][2:])*style.margin/100/2)
   if dilation:mask=cv2.dilate(mask,np.ones((2*dilation+1,2*dilation+1),np.uint8))
   if style.shape=='pixelise':mask=tiled_silhouette(mask,block=max(2,min(128,int((2+(style.tile_size*1.26 if style.tile_size is not None else style.intensity*.32))/2+.5)*2)) if style.effect in {'pixels_noirs','pixels_gris','pixels_rgb'} else max(4,round(4+style.intensity*.15)),dilation=0)
  if style.feather:
   radius=max(1,round(style.feather*.08));mask=cv2.GaussianBlur(mask,(radius*2+1,radius*2+1),0)
  mask=mask[y:y+bh,x:x+bw]
  alpha=mask[:,:,None].astype(np.float32)/255
  roi[:]=np.round(roi*(1-alpha)+masked*alpha).astype(np.uint8)
 if getattr(cfg,'brightness',100)!=100:out=np.clip(np.rint(out.astype(np.float32)*cfg.brightness/100),0,255).astype(np.uint8)
 return watermark(compose(out,cfg),cfg.watermark)

def rgb_color(value):
 from PIL import ImageColor
 return ImageColor.getrgb(value)

def watermark(frame,mark):
 if not mark.enabled or not mark.text:return frame
 rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).convert('RGBA');layer=Image.new('RGBA',rgb.size);draw=ImageDraw.Draw(layer)
 font=ImageFont.truetype(str(Path(__file__).resolve().parents[1]/'frontend/public/fonts/SFPRODISPLAYREGULAR.OTF'),mark.size)
 box=draw.textbbox((0,0),mark.text,font=font);tw,th=box[2]-box[0],box[3]-box[1];pad=round(min(rgb.size)*.035)
 x=pad if mark.position.endswith('left') else max(pad,rgb.width-tw-pad)
 y=pad if mark.position.startswith('top') else max(pad,rgb.height-th-pad)
 if mark.position=='center':x=(rgb.width-tw)//2;y=(rgb.height-th)//2
 alpha=round(mark.opacity*2.55);color=(*rgb_color(mark.color),alpha)
 if mark.banner:
  banner=mark.text.replace('\n',' ');bbox=draw.textbbox((0,0),banner,font=font);tw,th=bbox[2]-bbox[0],bbox[3]-bbox[1]
  step_x=max(1,tw+mark.size*2);step_y=max(1,th+mark.size*2)
  for row,yy in enumerate(range(pad,rgb.height,step_y)):
   for xx in range(pad,rgb.width,step_x):draw.text((xx,yy-bbox[1]),banner,font=font,fill=color)
  return cv2.cvtColor(np.array(Image.alpha_composite(rgb,layer).convert('RGB')),cv2.COLOR_RGB2BGR)
 draw.text((x,y-box[1]),mark.text,font=font,fill=color,stroke_width=1,stroke_fill=(0,0,0,alpha//2) if mark.color=='white' else (255,255,255,alpha//2))
 return cv2.cvtColor(np.array(Image.alpha_composite(rgb,layer).convert('RGB')),cv2.COLOR_RGB2BGR)

def compose(frame,cfg):
 from app import media
 h,w=frame.shape[:2];out=frame
 for image in cfg.image_overlays:
  original=cv2.imread(str(media(image.media_id)['path']),cv2.IMREAD_UNCHANGED)
  if original is None:raise ValueError('Image ajoutée illisible.')
  if original.ndim==2:original=cv2.cvtColor(original,cv2.COLOR_GRAY2BGRA)
  if original.shape[2]==3:original=cv2.cvtColor(original,cv2.COLOR_BGR2BGRA)
  width=max(1,round(w*image.width/100));height=max(1,round(h*image.height/100) if image.height is not None else round(original.shape[0]*width/original.shape[1]));tile=cv2.resize(original,(width,height));x,y=round(w*image.x/100),round(h*image.y/100);bw,bh=min(width,w-x),min(height,h-y)
  if bw<=0 or bh<=0:continue
  alpha=tile[:bh,:bw,3:4].astype(np.float32)/255*(image.opacity/100);out[y:y+bh,x:x+bw]=np.round(out[y:y+bh,x:x+bw]*(1-alpha)+tile[:bh,:bw,:3]*alpha).astype(np.uint8)
 for text in cfg.text_overlays:
  rgb=Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)).convert('RGBA');layer=Image.new('RGBA',rgb.size);draw=ImageDraw.Draw(layer)
  font=ImageFont.truetype(str(Path(__file__).resolve().parents[1]/'frontend/public/fonts/SFPRODISPLAYREGULAR.OTF'),text.size);alpha=round(text.opacity*2.55);color=(*rgb_color(text.color),alpha)
  draw.text((round(text.x*w/100),round(text.y*h/100)),text.text,font=font,fill=color,stroke_width=1,stroke_fill=(0,0,0,alpha//2) if text.color=='white' else (255,255,255,alpha//2))
  out=cv2.cvtColor(np.array(Image.alpha_composite(rgb,layer).convert('RGB')),cv2.COLOR_RGB2BGR)
 return out

# Existing imports keep their public API while detection is centrally orchestrated.
from vision_engine import PrivexVisionEngine as Detector
