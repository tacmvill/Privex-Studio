"""Repair generated head envelopes without changing native observations/identities."""
from collections import defaultdict
from bisect import bisect_left
import math

def repair(plan,media):
 if plan.get('head_envelope_revision')==1:return 0
 tracks=defaultdict(list);count=0
 for i,row in enumerate(plan['frames']):
  for obj in row:
   if obj.get('category',obj.get('class')) in {'tête','visage'}:tracks[obj['id']].append((i,obj))
 for ident,rows in tracks.items():
  trusted=[(i,o) for i,o in rows if o.get('detected') and o.get('model_source')!='Personne · tête estimée' and not o.get('identity_ambiguous')]
  times=[i for i,_ in trusted]
  for index,obj in rows:
   # Only repair known generated body/swept envelopes. User rectangles and
   # native detector boxes never get silently shrunk by this migration.
   if not (obj.get('coverage_evidence')=='unique-reidentification' or obj.get('render_envelope_guard') or obj.get('model_source')=='Personne · tête estimée'):continue
   at=bisect_left(times,index);neighbors=[r for r in trusted[max(0,at-1):at+1] if abs(r[0]-index)<=media['fps']*2]
   if len(neighbors)==2:
    a,left=neighbors[0];b,right=neighbors[1];t=(index-a)/max(1,b-a);box=[v+(z-v)*t for v,z in zip(left['box'],right['box'])]
   elif neighbors:box=list(neighbors[0][1]['box'])
   elif obj.get('body_evidence') and obj.get('model_source')=='Personne · tête estimée':
    from videomask.heads import person_to_head
    box=list(person_to_head(tuple(obj['body_evidence']['box']),media['width'],media['height']))
   else:continue
   x,y,w,h=obj['box'];nx,ny,nw,nh=box
   if w*h<=nw*nh*2.0 and h<=nh*1.7:continue
   if len(neighbors)==1:
    # Keep measured current center for modest drift; huge relays are usually
    # on the wrong body. Their nearest real head is the geometric reference.
    if math.hypot(x+w/2-nx-nw/2,y+h/2-ny-nh/2)<math.hypot(nw,nh)*1.5:nx,ny=x+w/2-nw/2,y+h/2-nh/2
   pad=.12;left,top=max(0,nx-nw*pad),max(0,ny-nh*pad);right,bottom=min(media['width'],nx+nw*(1+pad)),min(media['height'],ny+nh*(1+pad))
   if right<=left or bottom<=top:continue
   obj.update(box=[left,top,right-left,bottom-top],head_envelope_repaired=True,polygons=[],privacy_coverage=True);count+=1
 plan['head_envelope_revision']=1;plan['head_envelopes_repaired']=count
 return count
