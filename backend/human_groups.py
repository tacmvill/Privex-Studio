"""Use a continuously observed head as evidence linking body tracklets of one person."""
from collections import defaultdict,Counter
from proposals import same_zone

def merge(frames,fps,cuts=()):
 occurrences=defaultdict(dict);support=defaultdict(Counter);head_frames=defaultdict(set)
 for index,row in enumerate(frames):
  bodies=[o for o in row if o.get('category',o['class'])=='personne' and not o.get('auxiliary')]
  heads=[o for o in row if o.get('category',o['class']) in {'tête','visage'} and not o.get('identity_ambiguous')]
  for o in bodies:occurrences[o['id']][index]=o
  votes=[]
  for head in heads:
   head_frames[head['id']].add(index);x,y,w,h=head['box'];choices=[]
   for body in bodies:
    X,Y,W,H=body['box'];cx=x+w/2;cy=y+h/2
    if X<=cx<=X+W and Y<=cy<=Y+H*.4 and w<=W*1.1 and h<=H*.65:
     score=abs(cx-X-W/2)/max(1,W)+abs(cy-Y-H*.12)/max(1,H);choices.append((score,body['id']))
   choices.sort()
   if choices and (len(choices)==1 or choices[1][0]-choices[0][0]>.1):votes.append((choices[0][0],head['id'],choices[0][1]))
  by_body=defaultdict(list)
  for score,head,body in votes:by_body[body].append((score,head))
  for body,choices in by_body.items():
   choices.sort()
   if len(choices)==1 or choices[1][0]-choices[0][0]>.1:support[choices[0][1]][body]+=1
 parent={};owners={}
 def root(ident):
  while ident in parent:ident=parent[ident]
  return ident
 for head,counts in support.items():
  candidates=[ident for ident,n in counts.items() if n>=min(3,len(occurrences[ident])) and n/max(1,min(len(head_frames[head]),len(occurrences[ident])))>=.5]
  candidates.sort(key=lambda ident:min(occurrences[ident]))
  if not candidates:continue
  owner=root(candidates[0]);combined=dict(occurrences[owner])
  for ident in candidates[1:]:
   target=root(ident)
   if target==owner:continue
   current=occurrences[target];overlap=set(combined)&set(current)
   if overlap and any(not same_zone(combined[i]['box'],current[i]['box']) for i in overlap):continue
   a,b=sorted([max(combined),min(current)])
   if any(a<c<=b for c in cuts):continue
   if b-a>fps*2 and not all(i in head_frames[head] for i in range(a,b+1)):continue
   parent[target]=owner;combined.update(current)
  owners[head]=owner
  occurrences[owner]=combined
 for index,row in enumerate(frames):
  chosen={}
  for obj in row:
   ident=obj['id'];head=ident in owners;canonical=root(owners[ident]) if head else root(ident)
   obj=dict(obj)
   if canonical!=ident:
    obj.update(id=canonical,color=occurrences[canonical][min(occurrences[canonical])]['color'],reference_ids=list({ident,*obj.get('reference_ids',[])}))
    if head:obj.update(region_category=obj.get('category',obj['class']),category='personne',**{'class':'personne','class_id':0})
   old=chosen.get(canonical)
   if old is None or old.get('region_category') and not obj.get('region_category') or bool(old.get('region_category'))==bool(obj.get('region_category')) and obj.get('confidence',0)>old.get('confidence',0):chosen[canonical]=obj
  row[:]=chosen.values()
 return {'heads_associated':len(owners),'body_fragments_joined':len(parent)}
