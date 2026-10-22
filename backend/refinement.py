"""A second detection pass over discovery keyframes and representative difficult regions."""
import math
from videomask.persons import iou
from performance import phase,tick,timed

def refine(media,cfg,detector,frames,presence,first,last,job):
 if cfg.profile!='extreme' or not hasattr(detector,'deep'):return
 requests={};episodes={};period=max(1,round(media['fps']));representatives={}
 # Discovery sweeps remain available even when the coarse detectors found no human.
 for index in range(first,last):
  if cfg.analysis_ranges and not any(a<=index<b for a,b in cfg.analysis_ranges):continue
  if index==first or index in presence.cuts or (index-first)%period==0:requests[index]={'full':True,'regions':[]}
  humans=[o for o in frames[index] if o.get('detected') and o.get('category',o['class']) in {'tête','visage','personne'}]
  auxiliary=[b for b in presence.bodies.get(index,[]) if not any(o.get('category',o['class'])=='personne' and iou(tuple(o['box']),tuple(b['box']))>.6 for o in humans)]
  for obj in [*humans,*auxiliary]:
   x,y,w,h=obj['box'];cat=obj.get('category',obj['class']);has_head=any(o.get('category',o['class']) in {'tête','visage'} and x<=o['box'][0]+o['box'][2]/2<=x+w and y<=o['box'][1]+o['box'][3]/2<=y+h*.6 for o in humans)
   if cat=='personne' and has_head:continue
   if obj.get('confidence',0)>=.65 and w>=40:continue
   # One representative per stable episode; losses, motion or confidence drops open a new episode.
   ident=obj.get('id',f'body-{round(x/80)}-{round(y/80)}');previous=episodes.get(ident);episode=previous[2] if previous else 0
   if previous and (index-previous[0]>max(2,round(media['fps']*.3)) or iou(tuple(previous[1]),tuple(obj['box']))<.15 or obj.get('confidence',0)<previous[3]*.65):episode+=1
   episodes[ident]=(index,obj['box'],episode,obj.get('confidence',0));key=(ident,episode)
   score=obj.get('confidence',0);old=representatives.get(key)
   if old is None or score>old[0]:representatives[key]=(score,index,(max(0,x-w*.25),max(0,y-h*.2),w*1.5,h*(.8 if cat=='personne' and cfg.heads else 1.4)))
 for _,index,box in representatives.values():requests.setdefault(index,{'full':False,'regions':[]})['regions'].append(box)
 phase('Détection approfondie',len(requests));counter=0
 from inspection import source_at
 for done,(index,request) in enumerate(sorted(requests.items()),1):
  if job.get('cancel'):raise InterruptedError('Analyse approfondie annulée.')
  frame=source_at(media,index)
  with timed('detection/refinement'):found=detector.deep(frame,request['regions'],request['full'])
  presence.record(index,[*frames[index],*found])
  for obj in found:
   if obj.get('auxiliary') or obj['confidence']<cfg.confidence:continue
   cat=obj.get('category',obj['class']);matches=sorted([(max(iou(tuple(p['box']),tuple(obj['box'])),.6 if cat in {'tête','visage'} and __import__('identity_groups').head_overlap(p['box'],obj['box']) else 0.),n,p) for n,p in enumerate(frames[index]) if p.get('category',p['class'])==cat],reverse=True)
   if matches and matches[0][0]>.35:
    prior=matches[0][2];score=lambda p:.55 if p.get('confidence_estimated') else p.get('confidence',0)
    prior.setdefault('model_confidences',{}).update(obj.get('model_confidences',{}))
    if score(obj)>score(prior):prior.update({k:v for k,v in obj.items() if k not in {'id','color','reference_ids'}});prior['detection_confidence']=score(obj)
    continue
   neighbors=[]
   for i in range(max(first,index-period//2),min(last,index+period//2+1)):
    for prior in frames[i]:
     if prior.get('category',prior['class'])==cat and iou(tuple(prior['box']),tuple(obj['box']))>.5:neighbors.append(prior)
   ids={p['id'] for p in neighbors}
   if len(ids)==1:ident=neighbors[0]['id'];color=neighbors[0]['color']
   else:counter+=1;ident=f'a-refine-{counter}';color=__import__('engine').PALETTE[counter%8]
   obj.update(id=ident,color=color,visible=True,detected=True,tracking_confidence=1.,detection_confidence=.55 if obj.get('confidence_estimated') else obj['confidence'],timestamp=index/media['fps'],trajectory_id=ident,visibility_confidence=1.);frames[index].append(obj)
  tick(done)
