"""Recover isolated OpenCV seek/decode failures with the bundled FFmpeg decoder."""
import subprocess,json,math
from pathlib import Path
from collections import OrderedDict
import cv2,numpy as np,imageio_ffmpeg

def image(path,flags=cv2.IMREAD_COLOR):
 try:frame=cv2.imdecode(np.fromfile(path,dtype=np.uint8),flags)
 except (OSError,cv2.error):frame=None
 if frame is None:raise ValueError('Le fichier image ne peut pas être décodé. Réimportez une copie PNG ou JPEG valide.')
 return frame

_last_frames=OrderedDict()
_tails={}

def _remember(path,index,frame):
 _last_frames[path]=(index,frame);_last_frames.move_to_end(path)
 while len(_last_frames)>2:_last_frames.popitem(last=False)

def _decode(m,index,cap=None):
 owned=cap is None
 if owned:cap=cv2.VideoCapture(str(m['path']));cap.set(cv2.CAP_PROP_POS_FRAMES,index)
 try:ok,frame=cap.read()
 finally:
  if owned:cap.release()
 if ok:return frame
 # Timestamp seek independently recovers codecs that OpenCV cannot seek reliably.
 cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-ss',str(index/m['fps']),'-i',str(m['path']),'-frames:v','1','-f','image2pipe','-vcodec','png','pipe:1']
 try:result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));frame=cv2.imdecode(np.frombuffer(result.stdout,dtype=np.uint8),cv2.IMREAD_COLOR) if result.stdout else None
 except (subprocess.TimeoutExpired,cv2.error):frame=None
 if cap is not None and not owned:cap.set(cv2.CAP_PROP_POS_FRAMES,index+1)
 return frame

def frame_at(m,index,cap=None):
 if m['type']=='image':return image(m['path'])
 path=Path(m['path']);name=str(path.resolve());stat=path.stat();stamp=(stat.st_size,stat.st_mtime_ns,m['frames'])
 cache=path.with_suffix('.decode.json');entry=_tails.get(name)
 if entry is None or entry[0]!=stamp:
  _last_frames.pop(name,None);tail=None
  try:
   saved=json.loads(cache.read_text(encoding='utf-8'))
   if saved.get('stamp')==list(stamp) and max(1,m['frames']-max(2,math.ceil(m['fps']*.25)))<=int(saved['tail_start'])<m['frames']:tail=int(saved['tail_start'])
  except (OSError,ValueError,KeyError):pass
  entry=(stamp,tail);_tails[name]=entry
 tail=entry[1]
 if tail is not None and index>=tail:
  previous=_last_frames.get(name)
  frame=previous[1] if previous and previous[0]==tail-1 else _decode(m,tail-1)
  if frame is not None:
   m['tail_padding_frames']=m['frames']-tail;_remember(name,tail-1,frame);return frame
 frame=_decode(m,index,cap)
 if frame is not None:_remember(name,index,frame);return frame
 # Some containers report the longer audio duration as a video frame count.
 # A short, verified EOF follows normal player behaviour: hold the last real
 # frame. A hole with later readable pictures is still a decoding error.
 limit=max(2,math.ceil(m['fps']*.25));boundary=max(0,m['frames']-limit)
 if index>=boundary and all(_decode(m,i) is None for i in range(index+1,m['frames'])):
  previous=_last_frames.get(name)
  candidates=range(index-1,max(0,boundary-1)-1,-1)
  for last in candidates:
   found=previous[1] if previous and previous[0]==last else _decode(m,last)
   if found is None:continue
   tail=last+1;_tails[name]=(stamp,tail);_remember(name,last,found);m['tail_padding_frames']=m['frames']-tail
   tmp=cache.with_suffix('.decode.tmp')
   try:tmp.write_text(json.dumps({'stamp':list(stamp),'tail_start':tail}),encoding='utf-8');tmp.replace(cache)
   except OSError:pass # A read-only original still remains usable in memory.
   return found
 raise ValueError(f'Impossible de décoder l’image {index} à {index/m["fps"]:.2f} s du passage conservé. Vérifiez ce passage ou réimportez la vidéo.')

# Reused native frames for backward tracking and regional verification. 64 MiB cap.
_analysis_cache=None
class AnalysisFrames:
 def __init__(self,m):
  self.m=m;self.path=str(m['path']);self.items=OrderedDict();self.bytes=0;self.budget=64*1024**2;self.cap=cv2.VideoCapture(self.path);self.cursor=None
 def put(self,index,frame):
  if index in self.items:self.bytes-=self.items.pop(index).nbytes
  self.items[index]=frame;self.bytes+=frame.nbytes
  while self.bytes>self.budget and len(self.items)>1:self.bytes-=self.items.popitem(last=False)[1].nbytes
 def get(self,index):
  if index in self.items:self.items.move_to_end(index);return self.items[index]
  block=max(1,min(16,self.budget//max(1,self.m['width']*self.m['height']*3*2)));start=index//block*block;end=min(self.m['frames'],start+block);found=None
  if self.cursor!=start:self.cap.set(cv2.CAP_PROP_POS_FRAMES,start)
  for i in range(start,end):
   frame=_frame_at_uncached(self.m,i,self.cap);self.put(i,frame)
   if i==index:found=frame
  self.cursor=end;return found
 def close(self):self.cap.release();self.items.clear()

def set_analysis_cache(m):
 global _analysis_cache
 if _analysis_cache and _analysis_cache.path==str(m['path']):return
 clear_analysis_cache()
 if m['type']=='video':_analysis_cache=AnalysisFrames(m)
def clear_analysis_cache():
 global _analysis_cache
 if _analysis_cache:_analysis_cache.close();_analysis_cache=None
_frame_at_uncached=frame_at
def frame_at(m,index,cap=None):
 cache=_analysis_cache if _analysis_cache and _analysis_cache.path==str(m['path']) else None
 if cache and cap is None:return cache.get(index)
 frame=_frame_at_uncached(m,index,cap)
 if cache:cache.put(index,frame)
 return frame
