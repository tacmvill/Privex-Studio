"""Collapse duplicate boxes, preserving distinct nearby people and head-only detection."""
from videomask.persons import iou
def same_zone(a,b):
 x,y,w,h=a;X,Y,W,H=b;inter=max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y));small=max(1,min(w*h,W*H));ratio=small/max(1,max(w*h,W*H));distance=((x+w/2-X-W/2)**2+(y+h/2-Y-H/2)**2)**.5/max(1,min(w,h,W,H))
 return iou(tuple(a),tuple(b))>.55 or inter/small>.8 and ratio>.3 and distance<.4
def unique(rows,whole_people=False):
 result=[]
 for obj in sorted(rows,key=lambda o:o.get('confidence',0),reverse=True):
  cat=obj.get('category',obj['class']);prior=next((p for p in result if p.get('category',p['class'])==cat and same_zone(p['box'],obj['box'])),None)
  if prior:
   prior.setdefault('model_confidences',{}).update(obj.get('model_confidences',{}));continue
  result.append(dict(obj))
 if whole_people:
  people=[o for o in result if o.get('category',o['class'])=='personne' and not o.get('auxiliary') and o.get('confidence',0)>=.6]
  kept=[]
  for obj in result:
   if obj.get('category',obj['class']) in {'tête','visage'}:
    x,y,w,h=obj['box'];body=next((p for p in people if p['box'][0]<=x+w/2<=p['box'][0]+p['box'][2] and p['box'][1]<=y+h/2<=p['box'][1]+p['box'][3]*.5 and p['box'][2]>=w and p['box'][3]>=h*1.5),None)
    if body is not None:body.setdefault('head_regions',[]).append(obj['box']);continue
   kept.append(obj)
  result=kept
 return result
