"""Persistent trajectory aliases and conservative duplicate-box consolidation."""
from collections import defaultdict,Counter
from videomask.persons import iou

def head_overlap(a,b):
 x,y,w,h=a;X,Y,W,H=b
 inter=max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y))
 ratio=min(w*h,W*H)/max(1,max(w*h,W*H))
 return iou(tuple(a),tuple(b))>=.72 or (ratio>=.2 and inter/max(1,min(w*h,W*H))>=.85 and abs(x+w/2-X-W/2)<min(w,W)*.4 and abs(y-Y)<min(h,H)*.4)

def consolidate(frames):
 parent={};first={};colors={};observations=defaultdict(dict);support=Counter();members={}
 def root(k):
  while k in parent:k=parent[k]
  return k
 def join(a,b):
  a,b=root(a),root(b)
  if a==b:return False
  left,right=members.get(a,{a}),members.get(b,{b})
  for x in left:
   for y in right:
    for i in observations[x].keys()&observations[y].keys():
     one,two=observations[x][i],observations[y][i]
     if not one.get('detected',True) or not two.get('detected',True):continue
     if not one.get('visible',True) or not two.get('visible',True):continue
     human=one.get('category',one['class']) in {'tête','visage'} and two.get('category',two['class']) in {'tête','visage'}
     if not head_overlap(one['box'],two['box']) if human else iou(tuple(one['box']),tuple(two['box']))<.5:return False
  if (first.get(a,10**12),a)>(first.get(b,10**12),b):a,b=b,a
  parent[b]=a;members[a]=left|right
  return True
 for i,row in enumerate(frames):
  bodies=[o for o in row if o.get('category',o['class']) in {'personne','tête','visage'}]
  for o in row:
   first.setdefault(o['id'],i);colors.setdefault(o['id'],o.get('color'));observations[o['id']][i]=o
  for n,a in enumerate(bodies):
   for b in bodies[n+1:]:
    ca,cb=a.get('category',a['class']),b.get('category',b['class'])
    if ca!=cb and {ca,cb}!={'tête','visage'}:continue
    if (iou(tuple(a['box']),tuple(b['box']))>=.85 if ca=='personne' else head_overlap(a['box'],b['box'])):support[tuple(sorted([a['id'],b['id']]))]+=1
 # Known aliases are established by prior motion/ReID/head-body associations.
 for row in frames:
  for o in row:
   for alias in o.get('reference_ids',[]):join(o['id'],alias)
 duplicates=0
 for (a,b),count in support.items():
  overlap=set(observations[a])&set(observations[b])
  if count<3 or count/max(1,len(overlap))<.85:continue
  if any(not head_overlap(observations[a][i]['box'],observations[b][i]['box']) if observations[a][i].get('category',observations[a][i]['class']) in {'tête','visage'} else iou(tuple(observations[a][i]['box']),tuple(observations[b][i]['box']))<.5 for i in overlap):continue
  if root(a)!=root(b) and join(a,b):duplicates+=1
 groups=defaultdict(set)
 for ident in {*first,*parent,*parent.values()}:groups[root(ident)].add(ident)
 for row in frames:
  chosen={}
  for original in row:
   ident=root(original['id']);obj=dict(original,id=ident,color=colors.get(ident) or original.get('color'),reference_ids=sorted(groups[ident]-{ident}))
   old=chosen.get(ident)
   if old is None or (bool(old.get('region_category')),-old['box'][2]*old['box'][3])>(bool(obj.get('region_category')),-obj['box'][2]*obj['box'][3]):chosen[ident]=obj
  row[:]=chosen.values()
 return {'duplicate_trajectories_joined':duplicates,'identity_groups':len(groups)}
