"""Bounded inspection of at most three sampled frames. Never scans the video."""
from pathlib import Path
import json,base64
import cv2,numpy as np
from engine import Detector,CLASS_IDS,PALETTE,mask_frame

ROOT=Path(__file__).resolve().parent
from storage_paths import DATA
def load_inspection(ref,mid):
 import re
 if not re.fullmatch('[a-f0-9]{32}',ref):raise ValueError('Aperçu inconnu. Relancez la détection.')
 path=DATA/(ref+'.inspection.json')
 if not path.exists():raise ValueError('Aperçu absent. Relancez la détection.')
 data=json.loads(path.read_text(encoding='utf-8'))
 if data['media_id']!=mid:raise ValueError('Cet aperçu appartient à un autre média.')
 return data

def source_at(m,index):
 return __import__('video_io').frame_at(m,index)

def indices(m,times):
 if m['type']=='image':return [0]
 times=times or [m['duration']*f for f in [.15,.5,.85]]
 return [min(m['frames']-1,max(0,round(t*m['fps']))) for t in times[:3]]

def manual_on_sample(m,seed,index,frame,anchors=None):
 """Match an anchor directly to this sample; no intermediate-frame traversal."""
 if seed.anchors:
  from app import Seed
  nearest=min([*[a.model_dump() for a in reversed(seed.anchors)],{'time':seed.time,'box':seed.box}],key=lambda a:abs(a['time']-index/m['fps']))
  seed=Seed(id=seed.id,label=seed.label,time=nearest['time'],box=nearest['box'],smart=seed.smart,category=seed.category)
 anchor=min(m['frames']-1,round(seed.time*m['fps']))
 w,h=m['width'],m['height'];x,y,bw,bh=[round(v*n/100) for v,n in zip(seed.box,[w,h,w,h])]
 x=max(0,min(w-1,x));y=max(0,min(h-1,y));bw=min(bw,w-x);bh=min(bh,h-y)
 if bw<5 or bh<5:return None
 if anchor==index or m['type']=='image':return [x,y,bw,bh]
 if seed.smart:return None  # Never match the surrounding scenery across distant controls.
 original=anchors[anchor] if anchors is not None and anchor in anchors else source_at(m,anchor);template=cv2.cvtColor(original[y:y+bh,x:x+bw],cv2.COLOR_BGR2GRAY)
 if template.std()<8:return None  # uniform regions cannot be located reliably
 scale=min(1,960/max(w,h));target=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(round(w*scale),round(h*scale)))
 tile=cv2.resize(template,(max(5,round(bw*scale)),max(5,round(bh*scale))))
 if tile.shape[0]>target.shape[0] or tile.shape[1]>target.shape[1]:return None
 scores=cv2.matchTemplate(target,tile,cv2.TM_CCOEFF_NORMED);_,score,_,pos=cv2.minMaxLoc(scores)
 if score<.65:return None
 return [round(pos[0]/scale),round(pos[1]/scale),bw,bh]


def subject_key(ident):
 return ident[3:] if ident.startswith('m-da') else ident

def library_seeds(cfg):
 """Materialize only checked library entries; never invoke a detector."""
 from app import Seed
 excluded={subject_key(k) for k in cfg.excluded_ids};selected={subject_key(k) for k in cfg.selected_ids};seeds={};custom=set();virtual={subject_key(d.target_id) for d in cfg.temporal_decisions if d.action=='split'}
 for seed in cfg.custom:
  key=subject_key(seed.id);custom.add(key)
  if seed.hidden or key in excluded:continue
  seeds[key]=seed.model_copy(deep=True)
 for ref in cfg.references:
  key=subject_key(ref.id)
  if key in custom or key in virtual or key not in selected or key in excluded:continue
  override=cfg.subject_overrides.get(key,{})
  seeds[key]=Seed(id='m-d'+key if key.startswith('a') else key,time=ref.time,box=ref.box,label=override.get('class',ref.label),category=ref.category or ref.label,color=override.get('color',ref.color),style=override.get('style',ref.style),anchors=ref.anchors,reference_ids=ref.reference_ids)
 # A selected descendant still needs its source trajectory, even when the
 # source mask itself was unchecked. Rendering gates keep those roles separate.
 requested=(selected-excluded)|set(seeds)
 for _ in range(8):
  changed=False
  for decision in cfg.temporal_decisions:
   if decision.action not in {'split','link'} or subject_key(decision.target_id) not in requested:continue
   source=subject_key(decision.subject_id)
   if source not in requested:requested.add(source);changed=True
   if source in seeds:continue
   original=next((s for s in [*cfg.custom,*cfg.references] if subject_key(s.id)==source),None)
   if original is not None:
    seed=original.model_copy(deep=True);seed.id=decision.subject_id;seed.hidden=False;seed.tracking_only=True;seeds[source]=seed
  if not changed:break
 return list(seeds.values())

def library_config(cfg):
 seeds=library_seeds(cfg);allowed={subject_key(s.id) for s in seeds}
 masks=({subject_key(s.id) for s in seeds if not s.tracking_only}|{subject_key(k) for k in cfg.selected_ids})-{subject_key(k) for k in cfg.excluded_ids}
 return cfg.model_copy(update={'custom':seeds,'scope':'selection','selected_ids':sorted(masks),'references':[r for r in cfg.references if subject_key(r.id) in allowed|masks]},deep=True)

def build_library_inspection(mid,m,cfg,times,job):
 """Apply saved boxes to the controls without discovering additional subjects."""
 seeds=library_seeds(cfg);cached=None
 for ref in dict.fromkeys([cfg.selection_ref,cfg.inspection_ref]):
  if not ref:continue
  try:cached=load_inspection(ref,mid);break
  except ValueError:continue
 wanted=indices(m,times);anchors={i:source_at(m,i) for i in wanted};rows=[]
 custom={subject_key(s.id) for s in cfg.custom}
 for number,index in enumerate(wanted):
  if job.get('cancel'):raise InterruptedError('Application annulée.')
  job['phase']=f'Application aux sujets cochés · image {number+1}/{len(wanted)}'
  frame=anchors[index];row=next((r for r in cached['samples'] if r['frame']==index),None) if cached else None;objects=[]
  tracked=__import__('plan_store').frame(cached.get('plan_ref',''),index) if cached and cached.get('plan_ref') else None
  projected=__import__('cached_tracking').project([tracked],seeds,cfg)[0] if tracked is not None else None
  originals={o['id']:o for o in projected or []}
  for seed in seeds:
   key=subject_key(seed.id)
   original=originals.get(seed.id) if projected is not None else next((o for o in row['objects'] if subject_key(o['id'])==key or key in o.get('reference_ids',[])),None) if row else None
   explicit=key in custom and (m['type']=='image' or any(round(a.time*m['fps'])==index for a in [seed,*seed.anchors]))
   if seed.id.startswith(('a','m-da')) and not seed.anchors:
    box=original['box'] if original else None
   else:box=manual_on_sample(m,seed,index,frame,anchors) if explicit or not original else original['box']
   if box:
    obj=dict(original) if original and not explicit else {}
    obj.update(id=seed.id,reference_ids=[key] if key!=seed.id else [],box=box,**{'class':seed.label,'category':seed.category,'color':seed.color})
    objects.append(obj)
   else:job.setdefault('sample_unmatched',[]).append({'subject_id':seed.id,'frame':index})
  if cfg.segmentation=='sam2' and objects:objects=__import__('advanced').annotate(frame,objects)
  rows.append({'frame':index,'time':index/m['fps'],'objects':objects});job['progress']=round(100*(number+1)/len(wanted))
 result=save_inspection(job['id'],mid,m,rows,'library',cached.get('provider','Zones de la bibliothèque') if cached else 'Zones de la bibliothèque')
 if cached and cached.get('plan_ref'):
  result['plan_ref']=cached['plan_ref'];(DATA/(job['id']+'.inspection.json')).write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
 return result

def build_inspection(mid,m,cfg,times,all_objects,job):
 from performance import phase,tick
 cfg=cfg.model_copy(deep=True)
 if all_objects and cfg.discovery_all:cfg.faces=True;cfg.heads=True;cfg.ocr=True;cfg.brands=True;cfg.extended=__import__('extended').EXTENDED_NAMES;cfg.classes=list(CLASS_IDS);cfg.plates=True;cfg.scope='tous'
 phase('Chargement des modèles');detector=Detector(cfg);rows=[];wanted=indices(m,times)
 phase('Détection',len(wanted))
 for number,index in enumerate(wanted):
  if job.get('cancel'):raise InterruptedError('Analyse annulée.')
  job['phase']=f'Image de contrôle {number+1}/{len(indices(m,times))}'
  frame=source_at(m,index)
  objects=[d for d in detector.detect(frame) if d['confidence']>=cfg.confidence and not d.get('vehicle_continuation')]
  objects=[dict(d,id=f'a{index}-{d["class_id"]}-{i}',color=PALETTE[i%len(PALETTE)]) for i,d in enumerate(objects)]
  for seed in cfg.custom:
   if seed.hidden:continue
   box=manual_on_sample(m,seed,index,frame)
   if box:objects.append({'id':seed.id,'class':seed.label,'color':seed.color,'box':box})
   # A failed direct sample match is not a temporal tracking failure.
   # Keep it diagnostic only; audit/export compute actionable gaps frame by frame.
   else:job.setdefault('sample_unmatched',[]).append({'subject_id':seed.id,'frame':index})
  if cfg.segmentation=='sam2':objects=__import__('advanced').annotate(frame,objects)
  rows.append({'frame':index,'time':index/m['fps'],'objects':objects})
  tick(number+1,len(wanted))
 phase('Finalisation',1,'opérations')
 result=save_inspection(job['id'],mid,m,rows,'discover' if all_objects else 'analyze',detector.provider);tick(1,1)
 return result

def save_inspection(ref,mid,m,rows,mode,provider=''):
 for row in rows:
  for obj in row['objects']:obj.update(timestamp=row['time'],trajectory_id=obj['id'])
 data={'id':ref,'media_id':mid,'width':m['width'],'height':m['height'],'fps':m['fps'],'mode':mode,'provider':provider,'samples':rows}
 (DATA/(ref+'.inspection.json')).write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
 return data

def render_sample(data,row,cfg):
 from app import media,decorate
 m=media(data['media_id']);frame=source_at(m,row['frame']);objects=[dict(o) for o in row['objects']]
 # A newly drawn zone must appear immediately in a cached image preview.
 # On videos, only replace geometry at an explicit anchor; keep tracked boxes elsewhere.
 for seed in cfg.custom:
  if seed.id.startswith('m-da') or seed.id.startswith('a'):continue
  anchors=[seed,*seed.anchors]
  anchor=next((a for a in anchors if m['type']=='image' or round(a.time*m['fps'])==row['frame']),None)
  if anchor is None:continue
  objects=[o for o in objects if o['id']!=seed.id]
  if not seed.hidden:objects.append({'id':seed.id,'class':seed.label,'color':seed.color,'box':[round(v*n/100) for v,n in zip(anchor.box,[m['width'],m['height'],m['width'],m['height']])]})
 retained={s.id for s in cfg.custom if not s.hidden}
 if data['mode']=='discover' or (data['mode']=='analyze' and cfg.discovery_mode):objects=[o for o in objects if o['id'] in retained or o['id'] in cfg.selected_ids or any(k in cfg.selected_ids or 'm-d'+k in retained for k in o.get('reference_ids',[])) or (o['id'].startswith('m-da') and o['id'][3:] in cfg.selected_ids) or (not o['id'].startswith('a') and not o['id'].startswith('m-da'))]
 if data['mode']=='export' and cfg.discovery_mode:objects=[o for o in objects if o['id'] in retained or not o['id'].startswith('m-da') or o['id'][3:] in cfg.selected_ids]
 if cfg.library_only:
  allowed={subject_key(s.id) for s in library_seeds(cfg)}
  objects=[o for o in objects if subject_key(o['id']) in allowed or any(subject_key(k) in allowed for k in o.get('reference_ids',[]))]
 objects=decorate(objects,cfg,row['time'])
 if not cfg.preview_layers:cfg=cfg.model_copy(update={'brightness':100,'text_overlays':[],'image_overlays':[],'watermark':cfg.watermark.model_copy(update={'enabled':False})})
 result=mask_frame(frame,objects,cfg)
 return {'time':row['time'],'frame':row['frame'],'boxes':normalized(objects,m),'image':'data:image/png;base64,'+base64.b64encode(cv2.imencode('.png',result,[cv2.IMWRITE_PNG_COMPRESSION,2])[1]).decode()}

def normalized(objects,m):
 from videomask.core import _clip_box
 result=[]
 for o in objects:
  if not all(np.isfinite(v) for v in o['box']):continue
  clipped=_clip_box(tuple(o['box']),m['width'],m['height'])
  if clipped:result.append(dict(o,box=[v/n*100 for v,n in zip(clipped,[m['width'],m['height'],m['width'],m['height']])]))
 return result

def candidate_seeds(mid,cfg):
 from app import Seed,media
 if not cfg.discovery_mode:return []
 data=load_inspection(cfg.selection_ref or cfg.inspection_ref,mid);m=media(mid);kept=set(cfg.selected_ids);seeds=[]
 for row in data['samples']:
  for o in normalized(row['objects'],m):
   if o['id'] not in kept or not o['id'].startswith('a'):continue
   box=o['box']
   override=cfg.subject_overrides.get(o['id'],{})
   seeds.append(Seed(id='m-d'+o['id'],time=row['time'],box=box,label=override.get('class',o['class']),category=o.get('category',o['class']),color=override.get('color',o['color']),style=override.get('style')))
 existing={s.id[3:] for s in seeds}
 for ref in cfg.references:
  if ref.id in kept and ref.id not in existing and ref.id.startswith('a'):
   override=cfg.subject_overrides.get(ref.id,{})
   seeds.append(Seed(id='m-d'+ref.id,time=ref.time,box=ref.box,label=override.get('class',ref.label),category=ref.category,color=ref.color,style=override.get('style')))
 return seeds
