"""Shot-local association of fragmented trajectories using motion and visual galleries.

Features describe patches in this video, not biometric identities. Two simultaneous
distinct observations, cuts and non-unique matches prevent a join.
"""
from collections import defaultdict
from bisect import bisect_right,bisect_left
import math
import numpy as np,cv2
from videomask.persons import iou
from identity_groups import head_overlap

REVISION=1

def namespace_tracks(frames,gallery,token):
 """A fresh analysis must not reuse another analysis's internal subject IDs."""
 names={o['id'] for row in frames for o in row}|set(gallery)
 names|={alias for row in frames for o in row for alias in o.get('reference_ids',[])}
 mapping={name:f'a{token}-{name[1:]}' for name in names if name.startswith('a')}
 for row in frames:
  for obj in row:
   obj['id']=mapping.get(obj['id'],obj['id'])
   if 'trajectory_id' in obj:obj['trajectory_id']=mapping.get(obj['trajectory_id'],obj['trajectory_id'])
   obj['reference_ids']=[mapping.get(alias,alias) for alias in obj.get('reference_ids',[])]
 return {mapping.get(name,name):records for name,records in gallery.items()}

def remember(gallery,ident,index,obj,visual,body=None):
 if visual is None or obj.get('identity_ambiguous') or min(obj['box'][2:])<6:return
 record={'frame':index,'box':list(obj['box']),'visual':tuple(np.asarray(v,np.float32) for v in visual),'body':tuple(np.asarray(v,np.float32) for v in body) if body is not None else None}
 state=gallery.setdefault(ident,{})
 state.setdefault('first',record)
 if 'best' not in state or np.prod(obj['box'][2:])>np.prod(state['best']['box'][2:]):state['best']=record
 if 'last' in state:state['previous']=state['last']
 state['last']=record

def samples(gallery,ident,aliases):
 result=[];seen=set()
 for name in {ident,*aliases}:
  for record in gallery.get(name,{}).values():
   token=(name,record['frame'])
   if token not in seen:result.append(record);seen.add(token)
 return result

def similarity(first,second):
 scored=[];body=[]
 for a in first:
  for b in second:
   color=float(a['visual'][0]@b['visual'][0]);texture=max(0.,float(a['visual'][1]@b['visual'][1]))
   # Face/head detectors jitter and use different crop margins. Align small
   # central views before comparing texture, instead of declaring a new subject
   # whenever a face shifts a few pixels inside its detector rectangle.
   if color>=.55 and texture<.92 and a['visual'][1].size==576 and b['visual'][1].size==576:
    image=np.asarray(a['visual'][1],np.float32).reshape(24,24);other=np.asarray(b['visual'][1],np.float32).reshape(24,24)
    for size in [18,20]:
     pad=(24-size)//2;patch=image[pad:pad+size,pad:pad+size]
     if patch.std()>1e-3 and other.std()>1e-3:texture=max(texture,float(cv2.matchTemplate(other,patch,cv2.TM_CCOEFF_NORMED).max()))
   scored.append((.35*color+.65*texture,color,texture))
   if a.get('body') is not None and b.get('body') is not None:body.append(.4*float(a['body'][0]@b['body'][0])+.6*max(0.,float(a['body'][1]@b['body'][1])))
 scored.sort(reverse=True)
 # A changed pose may be compatible with only one of the stored views, but
 # plain matching skin colour without texture never establishes identity.
 return (scored[0] if scored else (0.,0.,0.)),max(body,default=0.)

def center(box):return np.asarray(box[:2],float)+np.asarray(box[2:],float)*.5
def kind(obj):
 category=obj.get('category',obj.get('class'))
 return 'head' if category in {'tête','visage'} else category

def stitch(frames,fps,cuts,gallery,job=None,max_gap=3.):
 tracks=defaultdict(dict);aliases=defaultdict(set)
 for index,row in enumerate(frames):
  for obj in row:
   aliases[obj['id']].update(obj.get('reference_ids',[]))
   if obj.get('detected',True) and obj.get('visible',True) and not obj.get('identity_ambiguous'):tracks[obj['id']][index]=obj
 entries={ident:sorted(rows.items()) for ident,rows in tracks.items()};starts=sorted((rows[0][0],ident) for ident,rows in entries.items());start_times=[i for i,_ in starts];candidates=[];tested=0
 features={ident:samples(gallery,ident,aliases[ident]) for ident in entries}
 def velocity(rows,last):
  rows=rows[-4:] if last else rows[:4]
  return np.median([(center(b['box'])-center(a['box']))/(j-i) for (i,a),(j,b) in zip(rows,rows[1:]) if j>i],axis=0) if len(rows)>1 else np.zeros(2)
 velocities={ident:(velocity(rows,True),velocity(rows,False)) for ident,rows in entries.items()}
 # Reliable common motion of other tracks contextualises a panning shot.
 by_frame={i:{obj['id']:obj for obj in row if obj.get('detected',True) and obj.get('visible',True) and not obj.get('identity_ambiguous')} for i,row in enumerate(frames)}
 duplicate_support=defaultdict(int)
 for row in by_frame.values():
  heads=[obj for obj in row.values() if kind(obj)=='head' and obj['id'].startswith('a')]
  for n,a in enumerate(heads):
   for b in heads[n+1:]:
    if head_overlap(a['box'],b['box']):duplicate_support[tuple(sorted([a['id'],b['id']]))]+=1
 for (a,b),support in duplicate_support.items():
  shared=tracks[a].keys()&tracks[b].keys()
  if support<min(2,len(tracks[a]),len(tracks[b])) or support/max(1,len(shared))<.9:continue
  if any(not head_overlap(tracks[a][i]['box'],tracks[b][i]['box']) for i in shared):continue
  if any(min(entries[a][0][0],entries[b][0][0])<cut<=max(entries[a][-1][0],entries[b][-1][0]) for cut in cuts):continue
  (visual,color,texture),body=similarity(features[a],features[b])
  if support==1 and (visual<.94 or texture<.75):continue
  # Repeated concentric face/head regions observe the same pixels. Native
  # face-only and whole-head crops need not have identical resized textures.
  if support>=2:visual=max(visual,.95)
  if (entries[a][0][0],a)>(entries[b][0][0],b):a,b=b,a
  candidates.append((visual,a,b,{'kind':'overlap','visual':round(visual,4),'body':round(body,4),'shared_observations':support}))
 def shared_motion(a,b):
  common=by_frame[a].keys()&by_frame[b].keys()
  if len(common)<5:return None
  motion=np.asarray([center(by_frame[b][k]['box'])-center(by_frame[a][k]['box']) for k in common]);median=np.median(motion,axis=0)
  sizes=[math.hypot(*by_frame[a][k]['box'][2:]) for k in common]
  return median if np.median(np.linalg.norm(motion-median,axis=1))<max(3,np.median(sizes)*.25) else None
 for before,rows in entries.items():
  if not before.startswith('a'):continue
  if job and job.get('cancel'):raise InterruptedError('Association des trajectoires annulée.')
  end,left=rows[-1];category=kind(left);limit=round(fps*max_gap)
  nearby=starts[bisect_left(start_times,end+1):bisect_right(start_times,end+limit)]
  for start,after in nearby:
   if before==after or not after.startswith('a'):continue
   other=entries[after];right=other[0][1]
   if category!=kind(right) or any(min(end,start)<c<=max(end,start) for c in cuts):continue
   shared=tracks[before].keys()&tracks[after].keys()
   # Co-visible different heads cannot be aliases, even with identical clothes.
   if shared and any(not head_overlap(tracks[before][i]['box'],tracks[after][i]['box']) if category=='head' else iou(tuple(tracks[before][i]['box']),tuple(tracks[after][i]['box']))<.65 for i in shared):continue
   if start<=end and (not shared or len(shared)<min(2,len(rows),len(other))):continue
   if not shared and (len(rows)<2 or len(other)<2):continue
   gap=max(1,start-end);seconds=gap/fps;old=np.asarray(left['box'],float);new=np.asarray(right['box'],float)
   ratio=min(np.prod(old[2:]),np.prod(new[2:]))/max(1,max(np.prod(old[2:]),np.prod(new[2:])))
   if ratio<.2:continue
   diagonal=max(8.,math.hypot(*old[2:]),math.hypot(*new[2:]));horizon=min(gap,fps*.8)
   forward=center(old)+velocities[before][0]*horizon;backward=center(new)-velocities[after][1]*horizon
   distance=max(np.linalg.norm(center(new)-forward),np.linalg.norm(center(old)-backward))/diagonal
   common=shared_motion(end,start) if start>end else None
   if common is not None:distance=min(distance,np.linalg.norm(center(new)-center(old)-common)/diagonal)
   if distance>.6+min(1.1,seconds*.4):continue
   tested+=1;(visual,color,texture),body=similarity(features[before],features[after])
   exact_geometry=seconds<=.12 and iou(tuple(old),tuple(new))>=.8 and ratio>=.65 and distance<.25 and visual>=.7 and texture>=.5
   strong=visual>=.87 and texture>=.75 and color>=.55
   contextual=body>=.9 and visual>=.58 and texture>=.35
   if not (strong or contextual or exact_geometry):continue
   if seconds>1.5 and visual<.92 and body<.94:continue
   appearance=max(visual,.6*visual+.4*body if contextual else 0.,.87 if exact_geometry else 0.)
   score=.72*appearance+.23*math.exp(-distance)+.05*ratio
   candidates.append((score,before,after,{'kind':'continuation','gap_frames':max(0,start-end-1),'visual':round(visual,4),'body':round(body,4),'motion_distance':round(float(distance),4),'shared_observations':len(shared)}))
 candidates.sort(reverse=True,key=lambda r:r[0]);parent={};members={k:{k} for k in entries};claimed_before=set();claimed_after=set();links=[]
 def root(k):
  while k in parent:k=parent[k]
  return k
 def conflict(a,b):
  for x in members[a]:
   for y in members[b]:
    shared=tracks[x].keys()&tracks[y].keys()
    if any(not head_overlap(tracks[x][i]['box'],tracks[y][i]['box']) if kind(tracks[x][i])=='head' else iou(tuple(tracks[x][i]['box']),tuple(tracks[y][i]['box']))<.65 for i in shared):return True
  return False
 outgoing=defaultdict(list);incoming=defaultdict(list)
 for score,before,after,evidence in candidates:
  if evidence['kind']=='continuation':outgoing[before].append((score,after));incoming[after].append((score,before))
 for score,before,after,evidence in candidates:
  if evidence['kind']=='continuation':
   competitors=[v for v,b in outgoing[before] if b!=after]+[v for v,a in incoming[after] if a!=before]
   if competitors and score-max(competitors)<.07 or before in claimed_before or after in claimed_after:continue
  a,b=root(before),root(after)
  if a==b or conflict(a,b):continue
  # Earliest ID is stable; all old IDs remain reversible selection aliases.
  if entries[a][0][0]>entries[b][0][0]:a,b=b,a
  parent[b]=a;members[a]|=members[b]
  if evidence['kind']=='continuation':claimed_before.add(before);claimed_after.add(after)
  links.append(dict(before=before,after=after,score=round(score,4),**evidence))
 groups={ident:members[root(ident)] for ident in entries}
 for row in frames:
  chosen={}
  for original in row:
   ident=root(original['id']) if original['id'] in entries else original['id'];group=groups.get(original['id'],{original['id']});all_aliases=set(original.get('reference_ids',[]))
   for name in group:all_aliases.update(aliases[name]);all_aliases.add(name)
   obj=dict(original,id=ident,reference_ids=sorted(all_aliases-{ident}))
   if ident in entries:obj['color']=entries[ident][0][1].get('color',obj.get('color'))
   old=chosen.get(ident)
   quality=lambda o:(bool(o.get('detected')),not o.get('geometry_estimated',False),o.get('detection_confidence',o.get('confidence',0)))
   if old is None or quality(obj)>quality(old):chosen[ident]=obj
  row[:]=chosen.values()
 return {'revision':REVISION,'tracks_before':len(entries),'tracks_after':len({root(k) for k in entries}),'linked_tracklets':len(links),'plausible_pairs':tested,'links':links}
