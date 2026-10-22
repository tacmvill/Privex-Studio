"""Project a completed detection/tracking plan onto a selection; no inference or CSRT."""
import json
from inspection import subject_key
FIELDS=('appearance_memory','profile','frame_step','heads','faces','classes','extended','ocr','brands','plates','grounding_prompts','confidence','tracker')
def signature(cfg):
 result={k:sorted(v) if isinstance(v,list) else v for k,v in cfg.model_dump(include=set(FIELDS)).items()}
 if not result.get('appearance_memory'):result.pop('appearance_memory',None)
 return result
def geometry(seed):return {'time':seed.time,'box':seed.box,'anchors':[a.model_dump() for a in seed.anchors],'smart':seed.smart,'category':seed.category}
def metadata(mid,cfg):
 from app import TEMPORAL_REVISION
 return {'media_id':mid,'revision':TEMPORAL_REVISION,'settings':signature(cfg),'custom':{subject_key(s.id):geometry(s) for s in cfg.custom if not s.hidden}}
def project(rows,seeds,cfg=None):
 lookup={}
 for seed in seeds:
  for k in {subject_key(seed.id),*[subject_key(r) for r in seed.reference_ids]}:lookup.setdefault(k,[]).append(seed)
 result=[]
 for row in rows:
  chosen={}
  for original in row:
   obj=__import__('linear_tracking').selected(original,cfg)
   if obj is None:continue
   targets={s.id:s for k in {subject_key(obj['id']),*[subject_key(v) for v in obj.get('reference_ids',[])]} for s in lookup.get(k,[])}
   for ident,seed in targets.items():
    if ident in chosen and chosen[ident].get('detection_confidence',0)>=obj.get('detection_confidence',0):continue
    chosen[ident]=dict(obj,id=ident,reference_ids=list({obj['id'],*obj.get('reference_ids',[]),*seed.reference_ids}),**{'class':seed.label,'category':seed.category or seed.label,'color':seed.color,'tracking_only':seed.tracking_only})
  result.append(list(chosen.values()))
 return result
def reuse(mid,cfg,job,frame_range=None):
 from app import DATA,TEMPORAL_REVISION,media
 from inspection import load_inspection,library_seeds
 seeds=library_seeds(cfg);m=media(mid)
 for ref in dict.fromkeys([cfg.selection_ref,cfg.inspection_ref]):
  if not ref:continue
  try:
   inspection=load_inspection(ref,mid);key=inspection.get('plan_ref','')
   if len(key)!=64:continue
   path=DATA/(key+'.plan.json')
   if not path.exists():continue
   source=json.loads(path.read_text(encoding='utf-8'));meta=source.get('analysis',{})
   if meta.get('revision')!=TEMPORAL_REVISION or meta.get('media_id')!=mid or meta.get('settings')!=signature(cfg):continue
   # A changed manual anchor needs another Detect pass; style/selection never does.
   if any(s.anchors and geometry(s)!=meta.get('custom',{}).get(subject_key(s.id)) or not s.id.startswith(('a','m-da')) and geometry(s)!=meta.get('custom',{}).get(subject_key(s.id)) for s in seeds):continue
   first,last=frame_range or (0,m['frames'])
   if not __import__('analysis_scope').contains(meta.get('coverage',[(0,m['frames'])]),first,last):continue
   __import__('linear_tracking').prepare(source,m['fps'])
   frames=project(source['frames'][first:last],seeds,cfg);frames=[[] for _ in range(first)]+frames+[[] for _ in range(m['frames']-last)]
   catalog={o['id']:dict(o,time=i/m['fps'],frame=i) for i,row in enumerate(frames) for o in row}
   if any(s.id not in catalog and s.id.startswith(('a','m-da')) and not any(subject_key(s.id) in {subject_key(o['id']),*[subject_key(v) for v in o.get('reference_ids',[])]} for o in source.get('catalog',[])) for s in seeds):continue
   mapping={}
   for row in frames:
    for o in row:
     for alias in o.get('reference_ids',[]):mapping.setdefault(subject_key(alias),set()).add(o['id'])
   review=[]
   for gap in source.get('review',[]):
    if gap['end']<=first/m['fps'] or gap['start']>=last/m['fps']:continue
    for ident in mapping.get(subject_key(gap['subject_id']),[]):
     seed=next(s for s in seeds if s.id==ident);a=max(first/m['fps'],gap['start']);b=min(last/m['fps'],gap['end']);review.append(dict(gap,id=f'{ident}:{round(a*m["fps"])}:{round(b*m["fps"])-1}:{gap["reason"]}',subject_id=ident,start=a,end=b,label=seed.label))
   result={**source,'frames':frames,'catalog':list(catalog.values()),'review':review,'reused_analysis':key};job['reused_analysis']=key;job['phase']='Suivi existant réutilisé · préparation du rendu';return result
  except (ValueError,OSError,KeyError):continue
 return None
