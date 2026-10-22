"""One disposable process per task; killing it interrupts native inference too."""
import sys,json,os,traceback,time
from pathlib import Path

def main():
 from app import Config,MEDIA,PLANS,analyze,export,fingerprint
 from inspection import build_inspection,load_inspection,save_inspection,indices,source_at,manual_on_sample,candidate_seeds
 payload=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));state_path=Path(payload['state_path'])
 if payload.get('data_dir'):
  __import__('app').DATA=__import__('inspection').DATA=__import__('plan_store').DATA=Path(payload['data_dir'])
 class Progress(dict):
  _write_lock=__import__('threading').RLock()
  def flush(self):
   with self._write_lock:self._flush()
  def _flush(self):
   tmp=state_path.with_suffix('.tmp');content=json.dumps(self,ensure_ascii=False);tmp.write_text(content,encoding='utf-8')
   # Windows may briefly deny replacing a file while the supervisor reads it.
   for _ in range(100):
    try:os.replace(tmp,state_path);return
    except PermissionError:time.sleep(.01)
   # Read/write sharing is allowed even when replacement sharing is denied.
   # The supervisor ignores an incomplete JSON document until the next poll.
   state_path.write_text(content,encoding='utf-8');tmp.unlink(missing_ok=True)
  def __setitem__(self,key,value):
   if key=='progress' and self.get('state')=='running':value=min(99,value)
   changed=self.get(key)!=value;super().__setitem__(key,value)
   if changed and key in ['progress','phase','performance']:self.flush()
 job=Progress(id=payload['id'],progress=0,cancel=False,warnings=[],state='running',phase='Préparation du moteur');job.flush()
 mid=payload['media_id'];m=payload['media'];m['path']=Path(m['path']);MEDIA[mid]=m;cfg=Config.model_validate(payload['settings']);kind=payload['kind']
 job['kind']=kind
 profiler=__import__('performance').Performance(job,__import__('app').DATA).begin()
 try:
  if kind=='montage':
   path=__import__('montage').export_sequence(payload['sequence'],job)
   if not path:raise RuntimeError('Le moteur n’a pas produit le rendu du montage.')
   job.update(state='done',progress=100);job.flush();return
  if kind in ['export','audit','scan']:
   job['phase']='Suivi et masquage de la vidéo entière'
   cfg=cfg.model_copy(deep=True)
   if kind=='scan':cfg=cfg.model_copy(update={'catalog_only':True,'library_only':False,'scope':'tous','discovery_mode':False,'custom':[s for s in cfg.custom if not s.id.startswith(('m-da','a')) or s.anchors],'excluded_ids':[]})
   if kind=='scan' and cfg.discovery_all:cfg=cfg.model_copy(update={'faces':True,'heads':True,'ocr':True,'brands':True,'extended':__import__('extended').EXTENDED_NAMES,'classes':list(__import__('engine').CLASS_IDS),'plates':True})
   elif cfg.library_only:cfg=__import__('inspection').library_config(cfg)
   elif cfg.discovery_mode:
    cfg.custom+= [s for s in candidate_seeds(mid,cfg) if s.id not in {x.id for x in cfg.custom}];cfg.scope='selection'
   if kind=='export':
    path=export(mid,cfg,job);job['output']=str(path)
   else:
    cached=__import__('app').DATA/(fingerprint(mid,cfg)+'.plan.json')
    cached_plan=json.loads(cached.read_text(encoding='utf-8')) if cached.exists() else None
    # Explicit detection refreshes obsolete retrospective analysis; exports keep their saved selection.
    if cached_plan is not None and (kind!='scan' or (not cfg.appearance_memory or cached_plan.get('appearance_memory_revision')==1) and cached_plan.get('vehicle_continuity_revision')==2 and cached_plan.get('context_identity_revision')==1 and cached_plan.get('playback_tracking_revision')==1 and cached_plan.get('retrospective_revision')==3 and (not (cfg.heads or cfg.faces) or cached_plan.get('head_detection_revision')==2)):PLANS[fingerprint(mid,cfg)]=cached_plan
    elif cfg.library_only and kind=='audit':
     existing=__import__('cached_tracking').reuse(mid,cfg,job)
     if existing is None:raise ValueError('Relancez Détecter pour préparer ce suivi. La vérification ne recalcule pas chaque sujet séparément.')
     PLANS[fingerprint(mid,cfg)]=existing;__import__('plan_store').save(fingerprint(mid,cfg),existing)
    else:analyze(mid,cfg,job)
    job['review']=__import__('temporal_intelligence').resolved_review(PLANS[fingerprint(mid,cfg)],cfg,m['fps'])
   plan=PLANS[fingerprint(mid,cfg)];rows=[{'frame':i,'time':i/m['fps'],'objects':plan['frames'][i]} for i in indices(m,payload['times'])]
   inspection=save_inspection(job['id'],mid,m,rows,'discover' if kind=='scan' else 'export',plan.get('provider',''))
   job['trajectories']=plan.get('trajectories',[]);job['catalog']=plan.get('catalog',[])
   inspection['plan_ref']=fingerprint(mid,cfg)
   (__import__('app').DATA/(job['id']+'.inspection.json')).write_text(json.dumps(inspection,ensure_ascii=False),encoding='utf-8')
  elif kind=='analyze' and cfg.library_only:
   __import__('contour_cache').prepare(mid,cfg,job,recover_onsets=True)
   inspection=__import__('inspection').build_library_inspection(mid,m,cfg,payload['times'],job)
  elif kind=='analyze' and cfg.discovery_mode:
   inspection=load_inspection(cfg.selection_ref or cfg.inspection_ref,mid);rows=[]
   wanted=indices(m,payload['times'])
   if [r['frame'] for r in inspection['samples']]!=wanted:raise ValueError('Les images de contrôle ont changé. Relancez « Détecter tous les éléments possibles ».')
   anchors={i:source_at(m,i) for i in set(wanted)};retained=candidate_seeds(mid,cfg)
   for number,row in enumerate(inspection['samples']):
    overridden={s.id[3:] for s in cfg.custom if s.id.startswith('m-da')}
    objects=[o for o in row['objects'] if o['id'].startswith('a') and o['id'] not in overridden];frame=anchors[row['frame']]
    for seed in [*cfg.custom,*[s for s in retained if s.id not in {z.id for z in cfg.custom}]]:
     if seed.hidden:continue
     if seed.id.startswith('m-da') and any(o['id']==seed.id[3:] for o in objects):continue
     box=manual_on_sample(m,seed,row['frame'],frame,anchors)
     if box:
      from videomask.persons import iou
      visible=[o for o in objects if o['id'] in cfg.selected_ids or not o['id'].startswith('a')]
      if any(o['class']==seed.label and iou(tuple(o['box']),tuple(box))>.9 for o in visible):continue
      objects.append({'id':seed.id,'class':seed.label,'color':seed.color,'box':box})
     # A failed direct sample match is not a temporal tracking failure.
     # Keep it diagnostic only; audit/export compute actionable gaps frame by frame.
     else:job.setdefault('sample_unmatched',[]).append({'subject_id':seed.id,'frame':row['frame']})
    if cfg.segmentation=='sam2':objects=__import__('advanced').annotate(frame,objects)
    rows.append(dict(row,objects=objects));job['progress']=round(100*(number+1)/len(wanted))
   inspection=save_inspection(job['id'],mid,m,rows,'discover',inspection['provider'])
  else:inspection=build_inspection(mid,m,cfg,payload['times'],kind=='discover',job)
  job.update(state='review' if kind=='audit' and job.get('review') else 'done',progress=100,inspection_ref=inspection['id'],images_analyzed=len(inspection['samples']),total_detected=sum(len(r['objects']) for r in inspection['samples']),provider=inspection['provider']);job.flush()
 except Exception as exc:
  job.update(state='error',error=str(exc));job.flush();traceback.print_exc()
 finally:profiler.finish();job.flush()
if __name__=='__main__':main()
