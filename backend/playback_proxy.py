"""On-demand, local browser-compatible copy. Original pixels/files remain export inputs."""
import hashlib,subprocess,threading,os
from pathlib import Path

_lock=threading.Lock()
_jobs={}

def prepare(mid):
 from app import media,DATA
 m=media(mid);source=Path(m['path']);stamp=f'{source.resolve()}:{source.stat().st_size}:{source.stat().st_mtime_ns}'
 key=hashlib.sha256(stamp.encode()).hexdigest();target=DATA/(key+'.playback.mp4')
 with _lock:
  if target.exists():return {'state':'ready','url':f'/api/media/{mid}/playback','source_unchanged':True}
  state=_jobs.get(key)
  if state:return dict(state)
  state={'state':'preparing','message':'Préparation d’une copie de lecture compatible…'};_jobs[key]=state
 def run():
  import imageio_ffmpeg
  partial=DATA/(key+'.playback.partial.mp4')
  try:
   args=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-y','-v','error','-i',str(source),'-map','0:v:0','-map','0:a:0?','-c:v','libx264','-preset','ultrafast','-crf','18','-threads','2','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(partial)]
   result=subprocess.run(args,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   if result.returncode:raise ValueError(result.stderr.decode(errors='replace')[-800:] or 'Échec du décodage vidéo.')
   os.replace(partial,target)
   with _lock:state.update(state='ready',url=f'/api/media/{mid}/playback')
  except Exception as exc:
   partial.unlink(missing_ok=True)
   with _lock:state.update(state='error',message='La copie de lecture n’a pas pu être préparée.',detail=str(exc))
 threading.Thread(target=run,daemon=True,name='privex-playback').start()
 return dict(state)

def path(mid):
 from app import media,DATA
 source=Path(media(mid)['path']);stamp=f'{source.resolve()}:{source.stat().st_size}:{source.stat().st_mtime_ns}'
 return DATA/(hashlib.sha256(stamp.encode()).hexdigest()+'.playback.mp4')
