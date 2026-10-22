"""Join mutually unique geometric continuations before expensive long-gap ReID."""
from collections import defaultdict
from bisect import bisect_right
from videomask.persons import iou
def stitch(frames,fps,cuts):
 tracks=defaultdict(list)
 for i,row in enumerate(frames):
  for o in row:tracks[o['id']].append((i,o))
 starts=sorted((v[0][0],k) for k,v in tracks.items());indices=[i for i,_ in starts];candidates=[];limit=max(1,round(fps*.25))
 for before,rows in tracks.items():
  end,last=rows[-1];x,y,w,h=last['box'];vx=vy=0
  if len(rows)>1:
   at,old=rows[-2];vx=(x-old['box'][0])/max(1,end-at);vy=(y-old['box'][1])/max(1,end-at)
  for start,after in starts[bisect_right(indices,end):bisect_right(indices,end+limit)]:
   head=tracks[after][0][1]
   if head.get('category',head['class'])!=last.get('category',last['class']) or any(end<c<=start for c in cuts):continue
   if last.get('identity_ambiguous') or head.get('identity_ambiguous'):continue
   gap=start-end;expected=(x+vx*gap,y+vy*gap,w,h);overlap=max(iou(tuple(last['box']),tuple(head['box'])),iou(expected,tuple(head['box'])))
   if last.get('category',last['class']) in {'tête','visage'} and __import__('identity_groups').head_overlap(last['box'],head['box']):overlap=max(overlap,.75)
   if overlap>=.55:candidates.append((overlap-.025*(gap-1),before,after))
 parent={};claimed_before=set();claimed_after=set()
 for score,before,after in sorted(candidates,reverse=True):
  competitors=[s for s,a,b in candidates if (a==before or b==after) and (a,b)!=(before,after)]
  if competitors and score-max(competitors)<.1 or before in claimed_before or after in claimed_after:continue
  root=before
  while root in parent:root=parent[root]
  parent[after]=root;claimed_before.add(before);claimed_after.add(after)
 for row in frames:
  for obj in row:
   original=obj['id'];ident=original
   while ident in parent:ident=parent[ident]
   if ident!=original:obj.update(id=ident,color=tracks[ident][0][1]['color'],reference_ids=list({original,*obj.get('reference_ids',[])}))
 return len(parent)
