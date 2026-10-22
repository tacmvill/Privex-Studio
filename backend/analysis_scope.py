"""Native source-frame ranges retained by the montage; no inference in removed passages."""
def ranges(media,requested,frame_range=None):
 raw=requested or ([frame_range] if frame_range else [(0,media['frames'])]);result=[]
 for a,b in sorted(raw):
  if a<0 or b<=a or a>=media['frames']:raise ValueError('Passage d’analyse invalide.')
  b=min(b,media['frames'])
  if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(b,result[-1][1]))
  else:result.append((a,b))
 return result
def contains(coverage,first,last):
 return any(a<=first and last<=b for a,b in coverage)
def review_in_scope(review,coverage,fps):
 result=[]
 for gap in review:
  for a,b in coverage:
   start=max(gap['start'],a/fps);end=min(gap['end'],b/fps)
   if end>start:result.append(dict(gap,start=start,end=end,id=f'{gap["subject_id"]}:{round(start*fps)}:{round(end*fps)-1}:{gap["reason"]}'))
 return result
