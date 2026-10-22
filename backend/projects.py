"""Validated, atomic local project files; sources stay in this installation."""
from pathlib import Path
from datetime import datetime,timezone
import json,uuid,os,threading,re
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field

router=APIRouter(prefix='/api/projects');lock=threading.Lock()
def root():
 from app import DATA
 folder=DATA/'projects';folder.mkdir(exist_ok=True);return folder
class ProjectRequest(BaseModel):
 format: str='videomask-project';version:int=1
 id:str|None=Field(None,pattern=r'^[a-f0-9]{32}$')
 name:str=Field('Mon projet',min_length=1,max_length=100)
 media_id:str=Field(pattern=r'^[a-f0-9]{32}$')
 settings:dict
 ui:dict=Field(default_factory=dict)
 stamp:int=Field(0,ge=0)

def validated(body):
 from app import Config,media
 if body.format!='videomask-project' or body.version!=1:raise HTTPException(400,'Format de projet non pris en charge.')
 try:cfg=Config.model_validate(body.settings)
 except Exception as e:raise HTTPException(400,'Les réglages du projet sont invalides.') from e
 media(body.media_id)
 for image in cfg.image_overlays:
  if not re.fullmatch('[a-f0-9]{32}',image.media_id):raise HTTPException(400,'Image ajoutée invalide.')
  media(image.media_id)
 candidates=body.ui.get('candidates',[])
 if not isinstance(candidates,list) or len(candidates)>2000:raise HTTPException(400,'Inventaire du projet invalide.')
 for candidate in candidates:
  if not isinstance(candidate,dict) or not isinstance(candidate.get('id'),str) or not isinstance(candidate.get('class'),str):raise HTTPException(400,'Sujet du projet invalide.')
  box=candidate.get('box')
  if not isinstance(box,list) or len(box)!=4 or any(not isinstance(n,(float,int)) or not 0<=n<=100 for n in box):raise HTTPException(400,'Hitbox du projet invalide.')
 if body.ui.get('sequence'):
  try:__import__('montage').Sequence.model_validate(body.ui['sequence']).validate_media(allow_empty=True)
  except ValueError as exc:raise HTTPException(400,'Montage invalide : '+str(exc))
 content=body.model_dump();content['settings']=cfg.model_dump();content['updated_at']=datetime.now(timezone.utc).isoformat()
 if len(json.dumps(content))>5*1024**2:raise HTTPException(400,'Projet trop volumineux.')
 return content
def write(path,content):
 temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 temp.write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)
def read(path):
 if not path.exists():raise HTTPException(404,'Projet introuvable.')
 try:return json.loads(path.read_text(encoding='utf-8'))
 except (ValueError,OSError) as e:raise HTTPException(400,'Fichier de projet illisible.') from e
def discard():
 path=root()/'draft.videomask'
 if path.exists():
  folder=root()/'recovery';folder.mkdir(exist_ok=True)
  os.replace(path,folder/(uuid.uuid4().hex+'.discarded.videomask'))

@router.get('')
def list_projects():
 items=[]
 for path in root().glob('*.videomask'):
  if not re.fullmatch('[a-f0-9]{32}',path.stem):continue
  try:data=read(path);items.append({k:data.get(k) for k in ['id','name','updated_at']})
  except HTTPException:continue
 return sorted(items,key=lambda p:p.get('updated_at') or '',reverse=True)
@router.get('/draft')
def get_draft():return read(root()/'draft.videomask')
@router.post('/draft')
def save_draft(body:ProjectRequest):
 content=validated(body)
 with lock:
  path=root()/'draft.videomask'
  if path.exists():
   previous=read(path)
   if previous.get('stamp',0)>body.stamp:return {'ok':True,'superseded':True}
  write(path,content)
 return {'ok':True}
@router.post('/draft/discard')
def discard_draft():
 with lock:discard()
 return {'ok':True}
@router.post('')
def save_project(body:ProjectRequest):
 content=validated(body);content['id']=body.id or uuid.uuid4().hex
 with lock:write(root()/(content['id']+'.videomask'),content)
 return {'id':content['id'],'name':content['name'],'updated_at':content['updated_at']}
@router.post('/import')
def import_project(body:ProjectRequest):
 content=validated(body)
 from app import media
 return {**content,'media':{k:v for k,v in media(body.media_id).items() if k!='path'}}
@router.get('/{id}/download')
def download_project(id:str):
 if not re.fullmatch('[a-f0-9]{32}',id):raise HTTPException(400,'Identifiant de projet invalide.')
 data=read(root()/(id+'.videomask'));name=re.sub(r'[^\w -]','_',data['name'])[:100]
 return FileResponse(root()/(id+'.videomask'),filename=name+'.videomask',media_type='application/json')
@router.get('/{id}')
def load_project(id:str):
 if not re.fullmatch('[a-f0-9]{32}',id):raise HTTPException(400,'Identifiant de projet invalide.')
 body=ProjectRequest.model_validate(read(root()/(id+'.videomask')))
 return import_project(body)
