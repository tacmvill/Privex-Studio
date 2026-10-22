from pathlib import Path
import json,uuid,threading,hashlib,subprocess,traceback,time,sys,re
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,imageio_ffmpeg
from fastapi import FastAPI,UploadFile,HTTPException,Request
from fastapi.responses import FileResponse,Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field,field_validator
from typing import Literal
from engine import Detector,MultiTracker,mask_frame,PALETTE,PROFILES
from videomask.custom_tracking import CustomObject,plan_custom_objects

ROOT=Path(__file__).resolve().parent
from storage_paths import DATA
TEMPORAL_REVISION='3.8-studio-1'
MEDIA={};JOBS={};PLANS={};PROCESSES={};JOB_LOCK=threading.Lock();PREVIEW_LIMIT_SECONDS=20.;pool=ThreadPoolExecutor(max_workers=1)
class Mark(BaseModel):
 enabled:bool=False;text:str=Field('',max_length=300);size:int=Field(32,ge=10,le=100);opacity:int=Field(70,ge=0,le=100);position:Literal['top-left','top-right','bottom-left','bottom-right','center']='bottom-right';color:str=Field('white',pattern=r'^(white|black|#[0-9a-fA-F]{6})$');banner:bool=False
class SubjectStyle(BaseModel):
 mosaic_color:str=Field('#007aff',pattern=r'^#[0-9a-fA-F]{6}$')
 tile_size:int|None=Field(None,ge=0,le=100)
 effect:Literal['flou','gaussien','noir','pixels','pixels_noirs','pixels_gris','pixels_rgb']='flou';shape:Literal['rectangle','lisse','pixelise']='rectangle'
 margin:int=Field(20,ge=0,le=100);intensity:int=Field(70,ge=0,le=100);size:int=Field(100,ge=50,le=150);feather:int=Field(0,ge=0,le=100)
def clipped_normalized_box(v):
 if not all(np.isfinite(n) for n in v) or v[2]<=0 or v[3]<=0:raise ValueError('Hitbox invalide : la zone doit avoir une taille positive.')
 x,y,w,h=v;left,top=max(0.,x),max(0.,y);right,bottom=min(100.,x+w),min(100.,y+h)
 if right<=left or bottom<=top:raise ValueError('La hitbox est entièrement hors de l’image.')
 return [left,top,right-left,bottom-top]
class Anchor(BaseModel):
 time:float=Field(0,ge=0);box:list[float]=Field(min_length=4,max_length=4)
 @field_validator('box')
 @classmethod
 def valid_anchor(cls,v):
  return clipped_normalized_box(v)
class Seed(BaseModel):
 smart:bool=False
 tracking_only:bool=False
 reference_ids:list[str]=[]
 id:str;time:float=Field(0,ge=0);box:list[float]=Field(min_length=4,max_length=4);label:str='Objet';category:str='';color:str='#0a84ff';hidden:bool=False;anchors:list[Anchor]=Field(default_factory=list,max_length=100);style:SubjectStyle|None=None
 @field_validator('box')
 @classmethod
 def valid_box(cls,v):
  return clipped_normalized_box(v)
class OverlayText(BaseModel):
 id:str;text:str=Field('Texte',max_length=300);x:float=Field(10,ge=0,le=100);y:float=Field(10,ge=0,le=100);size:int=Field(32,ge=10,le=150);opacity:int=Field(100,ge=0,le=100);color:str=Field('white',pattern=r'^(white|black|#[0-9a-fA-F]{6})$');banner:bool=False
class OverlayImage(BaseModel):
 id:str;media_id:str;height:float|None=Field(None,ge=1,le=100);x:float=Field(10,ge=0,le=100);y:float=Field(10,ge=0,le=100);width:float=Field(20,ge=1,le=100);opacity:int=Field(100,ge=0,le=100)
class TemporalDecision(BaseModel):
 id:str=Field(min_length=1,max_length=150);subject_id:str=Field(min_length=1,max_length=150)
 action:Literal['confirm','reject','suppress','absent','occluded','link','split','contour']
 start:float=Field(0,ge=0,le=21600,allow_inf_nan=False);end:float=Field(gt=0,le=21600,allow_inf_nan=False)
 target_id:str|None=Field(None,max_length=150)
 polygon:list[list[float]]|None=Field(None,min_length=3,max_length=128)
 @__import__('pydantic').model_validator(mode='after')
 def valid_interval(self):
  if self.end<=self.start:raise ValueError('Intervalle de décision invalide.')
  if self.action in {'link','split'} and (not self.target_id or self.target_id==self.subject_id):raise ValueError('Choisissez une autre identité cible.')
  if self.action=='contour' and (not self.polygon or any(len(p)!=2 or not all(np.isfinite(n) and 0<=n<=100 for n in p) for p in self.polygon)):raise ValueError('Contour invalide.')
  return self
class Config(BaseModel):
 appearance_memory:bool=False
 linear_tracking:bool=False
 catalog_only:bool=False
 evidence_classes:list[str]=[]
 temporal_decisions:list[TemporalDecision]=Field(default_factory=list,max_length=5000)
 analysis_ranges:list[tuple[int,int]]=Field(default_factory=list,max_length=300)
 frame_step:int=Field(1,ge=1,le=8)
 mosaic_color:str=Field('#007aff',pattern=r'^#[0-9a-fA-F]{6}$')
 tile_size:int|None=Field(None,ge=0,le=100)
 library_only:bool=False
 brightness:int=Field(100,ge=0,le=200)
 preview_layers:bool=True;discovery_all:bool=False
 # A detector-produced catalogue must round-trip intact, including crowded videos.
 grounding_prompts:list[str]=Field(default_factory=list,max_length=12);segmentation:Literal['yolo','sam2']='yolo';tracker:Literal['bytetrack','botsort','ocsort']='ocsort';references:list[Seed]=Field(default_factory=list);excluded_ids:list[str]=[];text_overlays:list[OverlayText]=Field(default_factory=list,max_length=30);image_overlays:list[OverlayImage]=Field(default_factory=list,max_length=20)
 inspection_ref:str='';selection_ref:str='';discovery_mode:bool=False;selected_ids:list[str]=Field(default_factory=list)
 heads:bool=True;extended:list[str]=[];ocr:bool=False;brands:bool=False;confirmed_gaps:list[str]=[];size:int=Field(100,ge=50,le=150);feather:int=Field(0,ge=0,le=100);subject_overrides:dict[str,dict]=Field(default_factory=dict)
 @__import__('pydantic').model_validator(mode='after')
 def identity_graph(self):
  from temporal_intelligence import key
  graph={}
  for d in self.temporal_decisions:
   if d.action in {'link','split'}:
    a,b=key(d.subject_id),key(d.target_id)
    if a==b:raise ValueError('Ces identifiants désignent déjà le même sujet.')
    graph.setdefault(a,set()).add(b)
  visiting=set();done=set()
  def visit(node):
   if node in visiting:raise ValueError('Ce rattachement crée une boucle entre les identités.')
   if node in done:return
   visiting.add(node)
   for target in graph.get(node,[]):visit(target)
   visiting.remove(node);done.add(node)
  for node in graph:visit(node)
  return self
 @field_validator('grounding_prompts')
 @classmethod
 def valid_prompts(cls,v):
  if any(len(n)>120 for n in v):raise ValueError('Description limitée à 120 caractères.')
  return list(dict.fromkeys(n.strip() for n in v if n.strip()))
 profile:Literal['rapide','moyen','extreme']='rapide';faces:bool=True;classes:list[str]=['personne'];plates:bool=False;scope:Literal['tous','selection']='tous';confidence:float=Field(.3,ge=.1,le=.9);margin:int=Field(20,ge=0,le=100);intensity:int=Field(70,ge=0,le=100);effect:Literal['flou','gaussien','noir','pixels','pixels_noirs','pixels_gris','pixels_rgb']='flou';shape:Literal['rectangle','lisse','pixelise']='rectangle';custom:list[Seed]=[];watermark:Mark=Mark();mute:bool=False;volume:int=Field(100,ge=0,le=100);quality:Literal['high','compact']='high'
 @field_validator('classes')
 @classmethod
 def valid_classes(cls,v):
  from engine import CLASS_IDS
  if any(n not in CLASS_IDS for n in v):raise ValueError('Classe inconnue.')
  return list(dict.fromkeys(v))
class JobRequest(BaseModel):
 media_id:str;settings:Config;sequence:dict|None=None;times:list[float]=Field(default_factory=list,max_length=3)
app=FastAPI(title='Privex Studio Local')
from system_controls import router as system_router
app.include_router(system_router)
@app.get('/api/categories')
def categories():
 from engine import COCO_NAMES
 from extended import EXTENDED_NAMES
 return {'objects':COCO_NAMES,'extended':EXTENDED_NAMES,'special':['visages','têtes','plaques','texte / marques écrites / heures']}
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])
@app.middleware('http')
async def local_only(request:Request,call_next):
 if request.headers.get('host','').split(':')[0] not in ['127.0.0.1','localhost','testserver']:return Response('Local only',403)
 if request.method=='POST' and request.headers.get('origin') not in [None,'http://127.0.0.1:5173','http://localhost:5173','http://127.0.0.1:8765']:return Response('Origin rejected',403)
 return await call_next(request)
def media(mid):
 if mid not in MEDIA and re.fullmatch('[a-f0-9]{32}',mid):
  sources=[p for p in DATA.glob(mid+'.*') if p.suffix.lower() in ['.mp4','.mov','.mkv','.avi','.webm','.m4v','.png','.jpg','.jpeg','.bmp','.webp']]
  if sources:
   p=sources[0];image=cv2.imread(str(p)) if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.webp'] else None
   if image is not None:h,w=image.shape[:2];fps=25.;frames=1;kind='image'
   else:
    cap=cv2.VideoCapture(str(p));w=int(cap.get(3));h=int(cap.get(4));fps=cap.get(5);frames=int(cap.get(7));cap.release();kind='video'
   if w and h and fps and frames:MEDIA[mid]={'id':mid,'path':p,'name':p.name,'type':kind,'width':w,'height':h,'fps':fps,'frames':frames,'duration':frames/fps if kind=='video' else 0}
 if mid not in MEDIA:raise HTTPException(404,'Média inconnu : réimportez le fichier.')
 return MEDIA[mid]
def frame_at(m,time):
 index=0 if m['type']=='image' else min(m['frames']-1,max(0,round(time*m['fps'])))
 return __import__('video_io').frame_at(m,index),index

def fingerprint(mid,cfg):
 content=cfg.model_dump(exclude={'appearance_memory','linear_tracking','brightness','preview_layers','watermark','mute','volume','quality','effect','mosaic_color','intensity','tile_size','shape','margin','inspection_ref','selection_ref','discovery_mode','selected_ids','subject_overrides','confirmed_gaps','size','feather','text_overlays','image_overlays','temporal_decisions'})
 virtual={__import__('temporal_intelligence').key(d.target_id) for d in cfg.temporal_decisions if d.action=='split'}
 content['references']=[r for r in content['references'] if __import__('temporal_intelligence').key(r['id']) not in virtual]
 if cfg.linear_tracking:content['linear_tracking']=1
 if cfg.appearance_memory:content['appearance_memory']=1
 content['engine_revision']=TEMPORAL_REVISION
 content['custom']=[{k:v for k,v in seed.items() if k not in ['label','color','style','tracking_only']} for seed in content['custom']]
 return hashlib.sha256((mid+json.dumps(content,sort_keys=True)).encode()).hexdigest()
def decorate(objects,cfg,time=0,labels=None):
 labels=labels if labels is not None else {s.id:s for s in (__import__('inspection').library_seeds(cfg) if cfg.library_only else cfg.custom)};result=[]
 for original in objects:
  obj=__import__('temporal_intelligence').resolve_object(original,cfg,time)
  if obj is None:continue
  seed=labels.get(obj['id']) or next((s for s in labels.values() if __import__('temporal_intelligence').key(s.id)==__import__('temporal_intelligence').key(obj['id'])),None) or next((s for s in labels.values() if s.id.removeprefix('m-d') in obj.get('reference_ids',[])),None);key=obj['id'][3:] if obj['id'].startswith('m-da') else obj['id'];keys=[key,*obj.get('reference_ids',[])]
  if seed and seed.hidden:continue
  if seed and seed.tracking_only and __import__('temporal_intelligence').key(obj['id']) not in {__import__('temporal_intelligence').key(k) for k in cfg.selected_ids}:continue
  if obj['id'] in cfg.excluded_ids or (not obj.get('source_subject_id') and any(k in cfg.excluded_ids for k in keys)):continue
  found=[cfg.subject_overrides[k] for k in keys if k in cfg.subject_overrides];override=cfg.subject_overrides.get(key) if obj.get('source_subject_id') and key in cfg.subject_overrides else max(found,key=lambda v:v.get('updated',0)) if found else {}
  obj['category']=obj.get('category',obj['class'])
  if seed:obj.update({'class':seed.label,'category':seed.category,'color':seed.color,'style':seed.style.model_dump() if seed.style else None})
  if override:obj.update({k:override[k] for k in ['class','color','style'] if k in override})
  if not any(o['id']==obj['id'] and __import__('videomask.persons',fromlist=['iou']).iou(tuple(o['box']),tuple(obj['box']))>.8 for o in result):result.append(obj)
 return result
@app.get('/api/status')
def status():return {'version':'2.9','temporal_revision':TEMPORAL_REVISION,'profiles':PROFILES,'models':{n:(ROOT/'models'/f).exists() for n,f in [('objets','yolo11n.onnx'),('visages','face_detection_yunet_2023mar.onnx'),('scrfd','det_10g.onnx'),('plaques','yolox_plates_s.onnx'),('têtes','head_detector.onnx'),('silhouettes','yolo11n-seg.onnx'),('vocabulaire ouvert','open-vocabulary.onnx')]},'advanced':__import__('advanced').installed(),'trackers':['OC-SORT + flux optique','ByteTrack','CSRT','BoT-SORT + ReID'],'local':True,'vision':__import__('vision_models').inventory()}
@app.post('/api/media')
async def upload(file:UploadFile):
 ext=Path(file.filename or '').suffix.lower()
 if ext not in ['.mp4','.mov','.mkv','.avi','.webm','.m4v','.png','.jpg','.jpeg','.bmp','.webp']:raise HTTPException(400,'Format de média non pris en charge.')
 mid=uuid.uuid4().hex;path=DATA/(mid+ext)
 with path.open('wb') as f:
  while chunk:=await file.read(1024*1024):f.write(chunk)
 image=cv2.imread(str(path)) if ext in ['.png','.jpg','.jpeg','.bmp','.webp'] else None
 if image is not None:h,w=image.shape[:2];fps=25.;frames=1;kind='image'
 else:
  cap=cv2.VideoCapture(str(path));w=int(cap.get(3));h=int(cap.get(4));fps=cap.get(5);frames=int(cap.get(7));cap.release();kind='video'
  if not w or not h or not fps or frames<1:raise HTTPException(400,'Le moteur ne peut pas décoder ce média.')
 m={'id':mid,'path':path,'name':file.filename,'type':kind,'width':w,'height':h,'fps':fps,'frames':frames,'duration':frames/fps if kind=='video' else 0};MEDIA[mid]=m
 return {k:v for k,v in m.items() if k!='path'}
@app.get('/api/media/{mid}/source')
def source(mid:str):return FileResponse(media(mid)['path'])
@app.post('/api/media/{mid}/playback')
def prepare_playback(mid:str):return __import__('playback_proxy').prepare(mid)
@app.get('/api/media/{mid}/playback')
def browser_playback(mid:str):
 path=__import__('playback_proxy').path(mid)
 if not path.exists():raise HTTPException(409,'La copie de lecture est en préparation.')
 return FileResponse(path,media_type='video/mp4')
@app.get('/api/media/{mid}/frame')
def source_frame(mid:str,time:float=0):
 frame,_=frame_at(media(mid),time);return Response(cv2.imencode('.png',frame,[cv2.IMWRITE_PNG_COMPRESSION,2])[1].tobytes(),media_type='image/png')
@app.get('/api/media/{mid}/crop')
def subject_crop(mid:str,time:float=0,x:float=0,y:float=0,w:float=100,h:float=100):
 if not all(np.isfinite(v) and 0<=v<=100 for v in [x,y,w,h]) or w<=0 or h<=0:raise HTTPException(400,'Zone invalide.')
 frame,_=frame_at(media(mid),max(0,time));height,width=frame.shape[:2]
 x1,y1=min(width-1,round(x*width/100)),min(height-1,round(y*height/100));x2,y2=min(width,round((x+w)*width/100)),min(height,round((y+h)*height/100))
 tile=frame[y1:max(y1+1,y2),x1:max(x1+1,x2)]
 return Response(cv2.imencode('.png',tile,[cv2.IMWRITE_PNG_COMPRESSION,2])[1].tobytes(),media_type='image/png',headers={'Cache-Control':'private, max-age=3600'})

@app.get('/api/inspection/{ref}/trajectories')
def trajectories(ref:str,mid:str):
 from inspection import load_inspection
 data=load_inspection(ref,mid);plan_ref=data.get('plan_ref','')
 if not re.fullmatch('[a-f0-9]{64}',plan_ref):return {'subjects':[],'review':[]}
 path=DATA/(plan_ref+'.plan.json')
 if not path.exists():raise HTTPException(409,'Relancez l’analyse globale.')
 plan=json.loads(path.read_text(encoding='utf-8'))
 return {'subjects':plan.get('trajectories',[]),'review':plan.get('review',[]),'scene_cuts':plan.get('scene_cuts',[]),'intelligence':plan.get('intelligence',{})}

@app.get('/api/inspection/{ref}/objects')
def inspection_objects(ref:str,mid:str,time:float=0):
 from inspection import load_inspection,normalized
 if not np.isfinite(time) or time<0:raise HTTPException(400,'Instant invalide.')
 try:data=load_inspection(ref,mid)
 except ValueError as exc:raise HTTPException(409,str(exc))
 m=media(mid);index=min(m['frames']-1,round(time*m['fps']));plan_ref=data.get('plan_ref','')
 row=__import__('plan_store').frame(plan_ref,index) if re.fullmatch('[a-f0-9]{64}',plan_ref) else None
 if row is None:row=next((r['objects'] for r in data['samples'] if r['frame']==index),[])
 return {'frame':index,'objects':normalized(row,m)}

@app.get('/api/inspection/{ref}/tracking-window')
def tracking_window(ref:str,mid:str,start:float=0,end:float=2):
 from inspection import load_inspection,normalized
 if not np.isfinite(start) or not np.isfinite(end) or start<0 or end<=start or end-start>5:raise HTTPException(400,'Fenêtre de suivi invalide.')
 try:data=load_inspection(ref,mid)
 except ValueError as exc:raise HTTPException(409,str(exc))
 m=media(mid);key=data.get('plan_ref','');first=max(0,int(start*m['fps']));last=min(m['frames'],int(end*m['fps'])+1,first+601)
 rows=__import__('plan_store').window(key,first,last) if re.fullmatch('[a-f0-9]{64}',key) else None
 if rows is not None:rows=[(i,[obj for original in row if (obj:=__import__('linear_tracking').selected(original)) is not None]) for i,row in rows]
 fields={'id','class','category','color','box','reference_ids','existence','presence','mask_allowed','visible','tracking_confidence'}
 return {'available':rows is not None,'fps':m['fps'],'frames':[{'frame':i,'objects':[{k:v for k,v in o.items() if k in fields} for o in normalized(row,m) if o.get('visible',True)]} for i,row in rows or []]}

@app.post('/api/inspection/{ref}/tracking-window')
def tracking_mask_window(ref:str,body:JobRequest,start:float=0,end:float=2):
 """Selected, decision-resolved masks for local playback. No detector calls."""
 from inspection import load_inspection,library_seeds,normalized
 m=media(body.media_id)
 try:data=load_inspection(ref,body.media_id)
 except ValueError as exc:raise HTTPException(409,str(exc))
 key=data.get('plan_ref','')
 from cached_tracking import signature,geometry
 meta=__import__('plan_store').metadata(key) if re.fullmatch('[a-f0-9]{64}',key) else {}
 seeds=library_seeds(body.settings)
 if meta.get('revision')!=TEMPORAL_REVISION or meta.get('media_id')!=body.media_id or meta.get('settings')!=signature(body.settings) or any((s.anchors or not s.id.startswith(('a','m-da'))) and geometry(s)!=meta.get('custom',{}).get(__import__('inspection').subject_key(s.id)) for s in seeds):
  raise HTTPException(409,'Relancez Détecter pour préparer le suivi avec ces réglages.')
 # Playback reads contours prepared by Apply; no segmentation, plan rewrite or duplicate SQLite read.
 if not np.isfinite(start) or not np.isfinite(end) or start<0 or end<=start or end-start>5:raise HTTPException(400,'Fenêtre de suivi invalide.')
 first=max(0,int(start*m['fps']));last=min(m['frames'],int(end*m['fps'])+1,first+601)
 native=__import__('plan_store').window(key,first,last) if re.fullmatch('[a-f0-9]{64}',key) else None
 box_fields={'id','class','category','color','box','reference_ids','existence','presence','mask_allowed','visible','tracking_confidence'}
 result={'available':native is not None,'fps':m['fps'],'frames':[{'frame':i,'objects':[{k:v for k,v in o.items() if k in box_fields} for o in normalized(row,m) if o.get('visible',True)]} for i,row in native or []]}
 projected=__import__('cached_tracking').project([row for _,row in native or []],seeds,body.settings);labels={s.id:s for s in seeds}
 masks={i:normalized(decorate(row,body.settings,i/m['fps'],labels=labels),m) for (i,_),row in zip(native or [],projected)}
 fields={'id','box','polygons','manual_contour','privacy_coverage','style','category','class'}
 coverage=meta.get('coverage',[(0,m['frames'])])
 for row,(_,original_row) in zip(result['frames'],native or []):
  visible_row=[obj for original in original_row if (obj:=__import__('linear_tracking').selected(original,body.settings)) is not None]
  row['objects']=[{k:v for k,v in o.items() if k in box_fields} for o in normalized(visible_row,m) if o.get('visible',True)]
  row['masks']=[{k:v for k,v in obj.items() if k in fields} for obj in masks.get(row['frame'],[])] if any(a<=row['frame']<b for a,b in coverage) else None
 return result

@app.get('/api/inspection/{ref}/catalog')
def inspection_catalog(ref:str,mid:str):
 from inspection import load_inspection,normalized
 try:data=load_inspection(ref,mid)
 except ValueError as exc:raise HTTPException(409,str(exc))
 m=media(mid);key=data.get('plan_ref','')
 catalog=__import__('plan_store').catalog(key) if re.fullmatch('[a-f0-9]{64}',key) else []
 candidates=[dict(o,sample=0) for o in normalized(catalog,m) if o['id'].startswith('a')] if catalog else [dict(o,time=row['time'],sample=i) for i,row in enumerate(data['samples']) for o in normalized(row['objects'],m) if o['id'].startswith('a')]
 return {'id':ref,'samples':[],'candidates':candidates,'mode':data['mode'],'provider':data['provider'],'total_masked':0}

@app.post('/api/inspection/{ref}/render')
def render_inspection(ref:str,body:JobRequest):
 from inspection import load_inspection,render_sample,normalized
 try:data=load_inspection(ref,body.media_id)
 except ValueError as exc:raise HTTPException(409,str(exc))
 samples=[render_sample(data,row,body.settings) for row in data['samples']]
 candidates=[dict(o,time=row['time'],sample=i) for i,row in enumerate(data['samples']) for o in normalized(row['objects'],media(body.media_id)) if o['id'].startswith('a')]
 if data.get('plan_ref') and re.fullmatch('[a-f0-9]{64}',data['plan_ref']):
  plan_path=DATA/(data['plan_ref']+'.plan.json')
  if plan_path.exists():
   all_objects=__import__('plan_store').catalog(data['plan_ref'])
   if all_objects:candidates=[dict(o,sample=0) for o in normalized(all_objects,data) if o['id'].startswith('a')]
 return {'id':ref,'samples':samples,'candidates':candidates,'mode':data['mode'],'provider':data['provider'],'total_masked':sum(len(row['boxes']) for row in samples)}

@app.post('/api/subjects/resolve')
def resolve_subject(body:JobRequest):
 if len(body.settings.custom)!=1:raise HTTPException(400,'Dessinez une seule zone à identifier.')
 try:return __import__('smart_selection').resolve(body.media_id,body.settings.custom[0],body.settings)
 except (ValueError,cv2.error) as exc:raise HTTPException(422,str(exc))

@app.post('/api/preview')
def preview(body:JobRequest,time:float=0):
 from inspection import load_inspection,render_sample
 try:data=load_inspection(body.settings.inspection_ref,body.media_id)
 except ValueError as exc:raise HTTPException(409,str(exc))
 m=media(body.media_id);index=min(m['frames']-1,max(0,round(time*m['fps'])))
 row=next((r for r in data['samples'] if r['frame']==index),None)
 if row is None and data.get('plan_ref') and re.fullmatch('[a-f0-9]{64}',data['plan_ref']):
  plan_path=DATA/(data['plan_ref']+'.plan.json')
  if plan_path.exists():
   objects=__import__('plan_store').frame(data['plan_ref'],index)
   if objects is not None:row={'frame':index,'time':index/m['fps'],'objects':objects}
 if row is None:raise HTTPException(409,'Cette image n’est pas analysée. Remplacez une image de contrôle par cet instant puis appliquez le traitement.')
 return render_sample(data,row,body.settings)

def vision_tracker(cfg,fps,job,namespace=''):
 if cfg.tracker=='botsort':
  try:
   inner=__import__('advanced').AppearanceTracker(fps,namespace=namespace);job['tracking_backend']='BoT-SORT + ReID'
   return __import__('identity_tracks').IdentityTracker(inner,fps,namespace,motion_fps=fps*cfg.frame_step,metrics=job.setdefault('online_tracking',{}))
  except Exception as exc:
   message=f'Suivi ReID indisponible ({type(exc).__name__}: {str(exc)[:180]}) : repli ByteTrack / CSRT.'
   if message not in job.setdefault('warnings',[]):job['warnings'].append(message)
 job['tracking_backend']='OC-SORT' if cfg.tracker=='ocsort' else 'ByteTrack'
 return __import__('identity_tracks').IdentityTracker(MultiTracker(fps,namespace=namespace,algorithm=cfg.tracker),fps,namespace,motion_fps=fps*cfg.frame_step,metrics=job.setdefault('online_tracking',{}))

@__import__('performance').profiled
def analyze(mid,cfg,job,frame_range=None):
 from performance import phase,tick,timed
 if cfg.library_only:cfg=__import__('inspection').library_config(cfg)
 m=media(mid);__import__('video_io').set_analysis_cache(m);__import__('temporal_intelligence').configure(m);coverage=__import__('analysis_scope').ranges(m,cfg.analysis_ranges,frame_range);first,last=coverage[0][0],coverage[-1][1];covered=lambda i:any(a<=i<b for a,b in coverage);starts={a for a,b in coverage};ends={b-1 for a,b in coverage}
 job['phase']='Détection et suivi · passages conservés' if frame_range or cfg.analysis_ranges else 'Détection et suivi · vidéo entière'
 if cfg.profile not in PROFILES:raise ValueError('Profil inconnu.')
 detector_cfg=cfg.model_copy(deep=True)
 if cfg.scope=='selection':
  categories={s.category or s.label for s in cfg.custom if not s.hidden}
  detector_cfg.scope='tous';detector_cfg.classes=list(set(cfg.classes)|{c for c in categories if c in __import__('engine').CLASS_IDS})
  if cfg.library_only:detector_cfg.classes=[c for c in categories if c in __import__('engine').CLASS_IDS];detector_cfg.faces=False;detector_cfg.heads=False;detector_cfg.extended=[];detector_cfg.ocr=False;detector_cfg.brands=False;detector_cfg.plates=False;detector_cfg.grounding_prompts=[]
  detector_cfg.faces=detector_cfg.faces or 'visage' in categories;detector_cfg.heads=detector_cfg.heads or bool(categories & {'tête','visage'});detector_cfg.extended=list(set(detector_cfg.extended)|{c for c in categories if c in __import__('extended').EXTENDED_NAMES});detector_cfg.ocr=detector_cfg.ocr or bool(categories & {'texte','heure lisible','texte de panneau'});detector_cfg.brands=detector_cfg.brands or any(c.startswith('marque ') for c in categories);detector_cfg.plates=detector_cfg.plates or 'plaque' in categories
 # Human bodies are evidence of possible presence even when a face is turned away.
 if detector_cfg.heads or detector_cfg.faces:detector_cfg.evidence_classes=['personne']
 detector_cfg.evidence_classes=list(set(detector_cfg.evidence_classes)|{s.category for s in cfg.custom if s.smart and not s.hidden and s.category in __import__('engine').CLASS_IDS})
 phase('Chargement des modèles');detector=Detector(detector_cfg);job['vision']=detector.report() if hasattr(detector,'report') else {};job.setdefault('warnings',[]).extend(job['vision'].get('warnings',[]));tracker=vision_tracker(cfg,m['fps']/cfg.frame_step,job);frames=[[] for _ in range(m['frames'])];cap=cv2.VideoCapture(str(m['path'])) if m['type']=='video' else None
 from temporal_presence import PresenceEvidence
 presence=PresenceEvidence(m,detector,first,last,job);presence.cuts=sorted({v for span in coverage for v in span if first<v<last});context_gallery={};tracker.context_gallery=context_gallery;phase('Détection',sum(b-a for a,b in coverage));processed=0
 try:
  if cap:cap.set(cv2.CAP_PROP_POS_FRAMES,first)
  followers=[]
  for index in range(first,last):
   if not covered(index):continue
   if cap and index in starts:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
   if job['cancel']:raise InterruptedError('Analyse annulée.')
   frame=__import__('video_io').frame_at(m,index,cap)
   cut=(presence.observe(index,frame) if cap else False) or index in starts and index!=first
   if cut:
    tracker=vision_tracker(cfg,m['fps']/cfg.frame_step,job,namespace=f's{index}-');tracker.context_gallery=context_gallery
    followers=[]
   if cut or (index-first)%cfg.frame_step==0 or index in ends:
    with timed('detection/coarse'):detected=detector.detect(frame)
    presence.record(index,detected)
    pending=[(o,o['id']) for o in getattr(tracker,'pending',[])]
    objects=[dict(d,continuation_only=d.get('continuation_only',False) or d['confidence']<cfg.confidence) for d in detected if d['confidence']>=min(.1,cfg.confidence) and not d.get('auxiliary')]
    if m['type']=='video':
     tracker.source_index=index;tracks=tracker.update(objects,frame,auxiliary=presence.bodies.get(index,[]))
    else:tracks=[dict(d,id=f'a{i}',color=PALETTE[i%len(PALETTE)]) for i,d in enumerate(objects) if not d.get('continuation_only')]
    # ByteTrack confirms provisional IDs on the next detector frame. Carry this
    # confirmation into the intervening CSRT frames as well.
    remapped={old:(o['id'],o['color']) for o,old in pending if old!=o['id']}
    if remapped:
     for prior in frames[max(first,index-cfg.frame_step):index]:
      for obj in prior:
       if obj['id'] in remapped:obj['id'],obj['color']=remapped[obj['id']]
   else:
    tracks=tracker.predict(frame)
   for obj in tracks:
    obj.setdefault('detected',True);obj.setdefault('detection_confidence',(.55 if obj.get('confidence_estimated') else obj.get('confidence',0)) if obj.get('detected') else 0.);obj.setdefault('tracking_confidence',1. if obj.get('detected') else 0.);obj.setdefault('visible',bool(obj.get('detected')))
   for obj in tracks:obj.update(timestamp=index/m['fps'],trajectory_id=obj['id'],visibility_confidence=1. if obj.get('detected') and obj.get('visible') else obj.get('tracking_confidence',0.) if obj.get('visible') else 0.)
   frames[index]=tracks;processed+=1;tick(processed)

 finally:
  if cap:cap.release()
 if m.get('tail_padding_frames'):job.setdefault('warnings',[]).append(f"Fin de piste vidéo : dernière image prolongée de {m['tail_padding_frames']/m['fps']:.2f} s pour correspondre à la durée du média.")
 __import__('refinement').refine(m,cfg,detector,frames,presence,first,last,job)
 context_token=hashlib.sha256(str(job.get('id') or uuid.uuid4().hex).encode()).hexdigest()[:8]
 context_gallery=__import__('context_association').namespace_tracks(frames,context_gallery,context_token)
 job['early_identity_groups']=__import__('identity_groups').consolidate(frames)
 phase('Association des sujets',1,'passages')
 with timed('tracking/context_association'):job['context_association']=__import__('context_association').stitch(frames,m['fps'],presence.cuts,context_gallery,job)
 tick(1)
 job['short_tracklets_joined']=job['context_association']['linked_tracklets']
 if cfg.catalog_only and m['type']=='video':__import__('printed_faces').reject(m,frames,presence.cuts,job,detector)
 if 'personne' in cfg.classes and (cfg.heads or cfg.faces):job['human_groups']=__import__('human_groups').merge(frames,m['fps'],presence.cuts)
 if m['type']=='video':job['flow_verified_gaps']=__import__('short_gaps').repair(m,frames,presence.cuts,job)
 from videomask.persons import iou
 links={}
 for ref in cfg.references:
  index=min(m['frames']-1,round(ref.time*m['fps']));box=tuple(v*n/100 for v,n in zip(ref.box,[m['width'],m['height'],m['width'],m['height']]))
  matches=[o for o in frames[index] if o.get('category',o['class'])==(ref.category or ref.label) and iou(tuple(o['box']),box)>.3]
  if matches:links.setdefault(max(matches,key=lambda o:iou(tuple(o['box']),box))['id'],[]).append(ref.id)
 for row in frames:
  for obj in row:obj['reference_ids']=list(set([*obj.get('reference_ids',[]),*links.get(obj['id'],[])]))
  row[:]=[o for o in row if o['id'] not in cfg.excluded_ids and not any(k in cfg.excluded_ids for k in o['reference_ids'])]
 catalog={}
 for index,row in enumerate(frames):
  for obj in row:
   if obj['id'] not in catalog:catalog[obj['id']]=dict(obj,time=index/m['fps'],frame=index)
 from tracking import track_subjects
 review=track_subjects(m,cfg,frames,job,(first,last),presence if cap else None)
 review=__import__('analysis_scope').review_in_scope(review,coverage,m['fps'])
 onset_stats=__import__('visible_onsets').recover(m,frames,presence.cuts,job,cfg,coverage)
 memory_stats=__import__('appearance_memory').recover(m,frames,presence.cuts,job,cfg,coverage)
 __import__('temporal_intelligence').stabilize(frames,presence.cuts)
 if cfg.segmentation=='sam2' and not cfg.catalog_only and (cfg.shape!='rectangle' or any(seed.style and seed.style.shape!='rectangle' for seed in cfg.custom)):
  job['phase']='Contours SAM 2 · vidéo entière'
  cap=cv2.VideoCapture(str(m['path'])) if m['type']=='video' else None
  try:
   if cap:cap.set(cv2.CAP_PROP_POS_FRAMES,first)
   for index in range(first,last):
    row=frames[index]
    if job['cancel']:raise InterruptedError('Segmentation annulée.')
    frame=__import__('video_io').frame_at(m,index,cap)
    if row:__import__('advanced').annotate(frame,row)
    job['progress']=round(100*(index+1)/m['frames'])
  finally:
   if cap:cap.release()
 if cfg.scope=='selection':
  selected={s.id for s in cfg.custom if not s.hidden}
  frames=[[o for o in row if o['id'] in selected] for row in frames]
 catalog={}
 for i,row in enumerate(frames):
  for o in row:
   previous=catalog.get(o['id'],{});aliases=list(set([*previous.get('reference_ids',[]),*o.get('reference_ids',[])]))
   if not previous or o.get('detection_confidence',0)>previous.get('detection_confidence',0):catalog[o['id']]=dict(o,time=i/m['fps'],frame=i)
   catalog[o['id']]['reference_ids']=aliases
   catalog[o['id']]['provisional']=catalog[o['id']].get('existence')=='candidate'
 phase('Finalisation',1,'opérations')
 result={'appearance_memory_revision':1,'appearance_memory':memory_stats,'vehicle_continuity_revision':2,'context_identity_revision':1,'context_association':job.get('context_association',{}),'playback_tracking_revision':1,'visible_onset_revision':1,'onset_checked_ids':onset_stats['checked_ids'],'visible_onsets':onset_stats,'head_detection_revision':2,'retrospective_revision':3,'entry_guard_version':2 if cfg.scope=='tous' else 0,'analysis':dict(__import__('cached_tracking').metadata(mid,cfg),coverage=coverage),'frames':frames,'provider':detector.provider,'review':review,'catalog':list(catalog.values()),'automatic_absences':presence.absences,'scene_cuts':presence.cuts,'trajectories':job.get('trajectories',[]),'intelligence':{'name':'Privex Temporal Intelligence','unresolved':len(review),'confirmed':sum(t['existence']=='confirmed' for t in job.get('trajectories',[])),'candidates':sum(t['existence']=='candidate' for t in job.get('trajectories',[])),'occlusions':sum(p['state']=='occluded' for t in job.get('trajectories',[]) for p in t['intervals']),'reidentified':job.get('automatic_reidentifications',0)}};__import__('render_continuity').repair(result);__import__('head_envelopes').repair(result,m);__import__('appearance_padding').prepare(result,m,cfg);__import__('linear_tracking').prepare(result,m['fps']);__import__('plan_store').save(plan_key(mid,cfg,frame_range),result);PLANS[plan_key(mid,cfg,frame_range)]=result
 tick(1,1)
 return result
def _clip_seed(box,m):
 from videomask.core import _clip_box
 result=_clip_box(box,m['width'],m['height'])
 if not result or result[2]<5 or result[3]<5:raise ValueError('Hitbox trop petite ou hors image.')
 return result
def plan_key(mid,cfg,frame_range=None):
 key=fingerprint(mid,cfg)
 return hashlib.sha256((key+str(frame_range)).encode()).hexdigest() if frame_range else key

def export(mid,cfg,job,frame_range=None):
 from performance import phase,tick
 if cfg.library_only:
  __import__('contour_cache').prepare(mid,cfg,job)
  cfg=__import__('inspection').library_config(cfg)
 m=media(mid);__import__('video_io').set_analysis_cache(m);__import__('temporal_intelligence').configure(m);first,last=frame_range or (0,m['frames']);key=plan_key(mid,cfg,frame_range);cached=DATA/(key+'.plan.json')
 plan=PLANS.get(key) or (json.loads(cached.read_text(encoding='utf-8')) if cached.exists() else None)
 if plan is None and cfg.library_only:plan=__import__('cached_tracking').reuse(mid,cfg,job,frame_range)
 if plan is None and cfg.library_only and m['type']=='video':raise ValueError('Le suivi de cette sélection doit être préparé avec Détecter. Relancez Détecter après une modification des cibles ou des marqueurs ; l’export réutilise ensuite ce suivi sans le recalculer.')
 if plan is None:plan=analyze(mid,cfg,job,frame_range)
 __import__('render_continuity').repair(plan)
 PLANS[key]=plan
 outstanding=__import__('temporal_intelligence').resolved_review(plan,cfg,m['fps'])
 if outstanding:
  # Advisory only: exporting does not require confirming presence or absence.
  # Keep the evidence unresolved; selected visible doubts are covered by
  # resolve_object, with the same explicit exclusions and rendering decisions.
  job.setdefault('export_uncertainties',[]).extend(dict(g,media_id=mid) for g in outstanding)
 path=DATA/(uuid.uuid4().hex+('.png' if m['type']=='image' else '.mp4'))
 phase('Finalisation',last-first)
 job['output']=str(path);job['phase']=('Encodage du passage conservé' if frame_range else 'Encodage de la vidéo entière') if m['type']=='video' else 'Enregistrement de l’image';job['progress']=0
 if m['type']=='image':frame,_=frame_at(m,0);cv2.imwrite(str(path),mask_frame(frame,decorate(plan['frames'][0],cfg,0),cfg));tick(1,1);return path
 exe=imageio_ffmpeg.get_ffmpeg_exe();crf='18' if cfg.quality=='high' else '23';cmd=[exe,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{m["width"]}x{m["height"]}','-r',str(m['fps']),'-i','pipe:0',*(['-ss',str(first/m['fps'])] if frame_range else []),'-i',str(m['path']),'-map','0:v:0']
 if not cfg.mute and cfg.volume:cmd+=['-map','1:a?','-c:a','aac','-af',f'volume={cfg.volume/100}']
 else:cmd+=['-an']
 if frame_range:cmd+=['-t',str((last-first)/m['fps'])]
 cmd+=['-c:v','libx264','-preset','veryfast','-crf',crf,'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
 err=DATA/(job['id']+'.ffmpeg.log');cap=cv2.VideoCapture(str(m['path']));cap.set(cv2.CAP_PROP_POS_FRAMES,first)
 with err.open('wb') as log:
  process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  try:
   for i in range(first,last):
    objects=plan['frames'][i]
    if job['cancel']:raise InterruptedError('Export annulé.')
    frame=__import__('video_io').frame_at(m,i,cap)
    if cfg.segmentation=='sam2' and (cfg.shape!='rectangle' or any(seed.style and seed.style.shape!='rectangle' for seed in cfg.custom)) and any(not obj.get('polygons') for obj in objects):objects=__import__('advanced').annotate(frame,[dict(o) for o in objects])
    process.stdin.write(mask_frame(frame,decorate(objects,cfg,i/m['fps']),cfg).tobytes());job['progress']=round(100*(i-first+1)/max(1,last-first))
    tick(i-first+1,last-first)
   process.stdin.close();code=process.wait(timeout=120)
   if code:raise RuntimeError(err.read_text(errors='replace')[-2000:])
  except BaseException:
   process.kill();process.wait();path.unlink(missing_ok=True);raise
  finally:cap.release()
 return path
def worker(job,request,kind):
 if job.get('cancel'):job.update(state='cancelled');return
 awake=__import__('keep_awake').KeepAwake();job['sleep_prevention']=awake.acquire()
 if awake.error:job.setdefault('warnings',[]).append(awake.error)
 process=None;state_path=DATA/(job['id']+'.task-state.json');payload_path=DATA/(job['id']+'.task-input.json');log_path=DATA/(job['id']+'.task.log')
 try:
  m=media(request.media_id);payload={'id':job['id'],'kind':kind,'data_dir':str(DATA),'media_id':request.media_id,'media':{k:str(v) if isinstance(v,Path) else v for k,v in m.items()},'settings':request.settings.model_dump(),'times':request.times,'sequence':request.sequence,'state_path':str(state_path)}
  payload_path.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
  with log_path.open('wb') as log:
   advanced=any(a.get('settings',{}).get('grounding_prompts') or a.get('settings',{}).get('segmentation')=='sam2' or a.get('settings',{}).get('tracker')=='botsort' for a in (request.sequence or {}).get('assets',[])) or bool(request.settings.grounding_prompts or request.settings.segmentation=='sam2' or request.settings.tracker=='botsort')
   executable=sys.executable
   if advanced and not getattr(sys,'frozen',False):
    candidates=[ROOT.parent/'.ai-venv/Scripts/python.exe',ROOT.parent/'.venv/Scripts/python.exe']
    executable=str(next((p for p in candidates if p.exists() and (p.parents[1]/'Lib/site-packages/torch').exists()),Path(sys.executable)))
   command=([sys.executable,'--worker',str(payload_path)] if getattr(sys,'frozen',False) else [executable,str(ROOT/'task_worker.py'),str(payload_path)])
   process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));PROCESSES[job['id']]=process
   while True:
    elapsed=time.monotonic()-job['started'];job['elapsed_seconds']=round(elapsed,1)
    if job['cancel']:
     _kill_task(process);job.update(state='cancelled',phase='Annulé',error='Traitement annulé.');break
    if kind not in ['export','audit','scan','montage'] and job.get('limit_seconds') is not None and elapsed>=job['limit_seconds']:
     _kill_task(process);job.update(state='error',phase='Délai dépassé',error='Aperçu interrompu : limite de calcul dépassée. Essayez le profil Rapide ou moins de cibles. Aucun traitement de la vidéo entière n’a été lancé.');break
    if state_path.exists():
     try:state=json.loads(state_path.read_text(encoding='utf-8'))
     except (OSError,json.JSONDecodeError):state={}
     job.update({k:v for k,v in state.items() if k not in ['id','state','cancel']})
     if state.get('state') in ['done','error','review']:
      job['state']=state['state'];break
    if process.poll() is not None:
     job.update(state='error',error='Le moteur de calcul s’est arrêté. Consultez le journal de la tâche.');break
    time.sleep(.06)
   if process.poll() is None:
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:_kill_task(process)
 except Exception as exc:job.update(state='error',error=str(exc));traceback.print_exc()
 finally:
  awake.release()
  if process and process.poll() is None:_kill_task(process)
  PROCESSES.pop(job['id'],None)
  if kind=='montage':
   import shutil
   folder=DATA/(job['id']+'.montage')
   if folder.resolve().parent==DATA.resolve():shutil.rmtree(folder,ignore_errors=True)
   protected={media(a['media_id'])['path'].resolve() for a in (request.sequence or {}).get('assets',[])}
   artifacts=[*job.get('montage_outputs',[]),*([job['montage_final']] if job.get('montage_final') and job['state']!='done' else [])]
   for value in artifacts:
    target=Path(value).resolve()
    if target.parent==DATA.resolve() and re.fullmatch(r'[a-f0-9]{32}\.(mp4|png)',target.name) and target not in protected:target.unlink(missing_ok=True)
  if job['state']!='done':
   # Remove only this worker's incomplete artifact, never imported media.
   if state_path.exists():
    try:
     state=json.loads(state_path.read_text(encoding='utf-8'));output=state.get('output')
     if output and Path(output).resolve().parent==DATA.resolve():Path(output).unlink(missing_ok=True)
    except (OSError,json.JSONDecodeError):pass

def _kill_task(process):
 if process.poll() is not None:return
 import psutil
 try:
  children=psutil.Process(process.pid).children(recursive=True)
  for child in children:
   try:child.kill()
   except psutil.NoSuchProcess:pass
  process.kill();process.wait(timeout=2)
 except (psutil.NoSuchProcess,ProcessLookupError):pass

@app.post('/api/jobs/{kind}')
def start(kind:str,body:JobRequest):
 media(body.media_id)
 if kind not in ['analyze','discover','export','audit','scan','montage']:raise HTTPException(400,'Tâche inconnue.')
 if kind=='montage':
  try:__import__('montage').Sequence.model_validate(body.sequence).validate_media()
  except ValueError as exc:raise HTTPException(422,str(exc))
 if any(not np.isfinite(t) or t<0 for t in body.times):raise HTTPException(422,'Instants de contrôle invalides.')
 with JOB_LOCK:
  if any(j['state']=='running' for j in JOBS.values()):raise HTTPException(409,'Une tâche est déjà en cours. Annulez-la avant de recommencer.')
  limit=180. if body.settings.profile=='extreme' else 90. if body.settings.grounding_prompts or body.settings.segmentation=='sam2' else PREVIEW_LIMIT_SECONDS
  if kind=='analyze' and body.settings.library_only and media(body.media_id)['type']=='video':limit=max(limit,min(600.,max(120.,media(body.media_id)['duration']*.5)))
  full_apply=kind=='analyze' and body.settings.library_only and media(body.media_id)['type']=='video'
  jid=uuid.uuid4().hex;job={'id':jid,'state':'running','kind':kind,'progress':0,'cancel':False,'warnings':[],'started':time.monotonic(),'phase':'Préparation du montage' if kind=='montage' else 'Préparation du média' if kind not in ['export','audit','scan'] else 'Détection des passages conservés' if kind=='scan' and body.settings.analysis_ranges else 'Traitement de la vidéo entière','limit_seconds':limit if kind not in ['export','audit','scan','montage'] and not full_apply else None};JOBS[jid]=job;pool.submit(worker,job,body,kind)
 return {'id':jid,'limit_seconds':job['limit_seconds']}
@app.get('/api/jobs/{jid}')
def get_job(jid:str):
 if jid not in JOBS:raise HTTPException(404,'Tâche inconnue.')
 return {k:v for k,v in JOBS[jid].items() if k not in ['cancel','output','started','montage_outputs','montage_final']}
@app.get('/api/jobs/{jid}/preview')
def preview_job(jid:str):
 job=JOBS.get(jid)
 if not job or job.get('state')!='done' or not job.get('output'):raise HTTPException(409,'L’aperçu complet n’est pas prêt.')
 return FileResponse(job['output'],media_type='video/mp4')

@app.post('/api/jobs/{jid}/cancel')
def cancel(jid:str):
 if jid not in JOBS:raise HTTPException(404,'Tâche inconnue.')
 if JOBS[jid]['state']!='running':return {'ok':False,'state':JOBS[jid]['state']}
 JOBS[jid]['cancel']=True;JOBS[jid]['phase']='Annulation en cours';process=PROCESSES.get(jid)
 if process:_kill_task(process)
 return {'ok':True}
@app.get('/api/jobs/{jid}/download')
def download(jid:str):
 job=JOBS.get(jid,{})
 if job.get('state')!='done' or not job.get('output'):raise HTTPException(409,'Export indisponible.')
 return FileResponse(job['output'],filename='Privex-Studio-export'+Path(job['output']).suffix)
from montage import router as montage_router
app.include_router(montage_router)
from projects import router as projects_router
app.include_router(projects_router)
DIST=ROOT.parent/'frontend/dist'
if DIST.exists():app.mount('/',StaticFiles(directory=DIST,html=True),name='studio')
