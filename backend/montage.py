"""Non-destructive two-track editing, cached audio peaks and frame-accurate FFmpeg assembly."""
from pathlib import Path
import json,math,subprocess,time,uuid,shutil,threading
import numpy as np
from pydantic import BaseModel,Field,model_validator
from fastapi import APIRouter,HTTPException
import imageio_ffmpeg
router=APIRouter(prefix='/api/montage')
class Asset(BaseModel):
 media_id:str=Field(pattern=r'^[a-f0-9]{32}$');name:str=Field(max_length=250);settings:dict=Field(default_factory=dict);project:dict=Field(default_factory=dict)
class Clip(BaseModel):
 id:str=Field(min_length=1,max_length=100);media_id:str=Field(pattern=r'^[a-f0-9]{32}$');track:str;start:float=Field(ge=0,le=21600,allow_inf_nan=False);source_in:float=Field(ge=0,le=21600,allow_inf_nan=False);source_out:float=Field(gt=0,le=21600,allow_inf_nan=False);volume:int=Field(100,ge=0,le=200);muted:bool=False;source_audio_override:bool=False;link:str|None=None
 @model_validator(mode='after')
 def interval(self):
  if self.track not in ['video','audio'] or self.source_out<=self.source_in:raise ValueError('Segment ou piste invalide.')
  if self.start+self.source_out-self.source_in>21600:raise ValueError('Montage limité à six heures.')
  return self
class TimedText(BaseModel):
 id:str=Field(min_length=1,max_length=100);text:str=Field('Texte',max_length=300);x:float=Field(10,ge=0,le=100,allow_inf_nan=False);y:float=Field(10,ge=0,le=100,allow_inf_nan=False);size:int=Field(32,ge=10,le=150);opacity:int=Field(100,ge=0,le=100);color:str=Field('white',pattern=r'^(white|black|#[0-9a-fA-F]{6})$');start:float=Field(ge=0,le=21600,allow_inf_nan=False);end:float=Field(gt=0,le=21600,allow_inf_nan=False)
 @model_validator(mode='after')
 def interval(self):
  if self.end<=self.start:raise ValueError('La fin du texte doit suivre son début.')
  return self
class TimedImage(BaseModel):
 id:str=Field(min_length=1,max_length=100);media_id:str=Field(pattern=r'^[a-f0-9]{32}$');name:str=Field('Image',max_length=250);x:float=Field(10,ge=0,le=100,allow_inf_nan=False);y:float=Field(10,ge=0,le=100,allow_inf_nan=False);width:float=Field(20,ge=1,le=100,allow_inf_nan=False);height:float|None=Field(None,ge=1,le=100,allow_inf_nan=False);opacity:int=Field(100,ge=0,le=100);start:float=Field(ge=0,le=21600,allow_inf_nan=False);end:float=Field(gt=0,le=21600,allow_inf_nan=False)
 @model_validator(mode='after')
 def interval(self):
  if self.end<=self.start:raise ValueError('La fin de l’image doit suivre son début.')
  return self
class Sequence(BaseModel):
 frame_step:int=Field(1,ge=1,le=8)
 assets:list[Asset]=Field(min_length=1,max_length=50);clips:list[Clip]=Field(default_factory=list,max_length=300);texts:list[TimedText]=Field(default_factory=list,max_length=30);images:list[TimedImage]=Field(default_factory=list,max_length=20)
 def validate_media(self,allow_empty=False):
  from app import media,Config
  known={a.media_id:a for a in self.assets}
  ids=[z.id for z in [*self.texts,*self.images]]
  if len(set(ids))!=len(ids):raise ValueError('Identifiants d’ajouts dupliqués.')
  for image in self.images:
   if media(image.media_id)['type']!='image':raise ValueError('Une superposition doit être une image.')
  if len(known)!=len(self.assets):raise ValueError('Médias dupliqués.')
  for asset in self.assets:media(asset.media_id);Config.model_validate(asset.settings)
  base=media(self.assets[0].media_id);project_fps=max(.001,base['fps']) if base['type']=='video' else 25
  for clip in self.clips:
   if clip.source_out-clip.source_in<1/project_fps-1e-6:raise ValueError('Un segment doit conserver au moins une image du montage.')
   if clip.track=='video' and any(abs(t*project_fps-round(t*project_fps))>.001 for t in [clip.start,clip.source_out-clip.source_in]):raise ValueError('Les coupes vidéo doivent être alignées sur les images du montage.')
   if clip.media_id not in known:raise ValueError('Média du segment absent.')
   m=media(clip.media_id)
   if m['type']=='image' and clip.track=='audio':raise ValueError('Une image ne possède pas de son.')
   if m['type']=='video' and clip.source_out>m['duration']+1/m['fps']:raise ValueError('La coupe dépasse la durée de la source.')
  visual=sorted([c for c in self.clips if c.track=='video'],key=lambda c:c.start)
  end=0
  for c in visual:
   if c.start<end-.001:raise ValueError('Les clips vidéo ne peuvent pas se chevaucher sur V1.')
   end=c.start+c.source_out-c.source_in
  if not self.clips and not self.texts and not self.images and not allow_empty:raise ValueError('Ajoutez au moins un segment au montage.')
  return self

_wave_locks={};_wave_guard=threading.Lock()
def waveform(mid,bins=4096):
 with _wave_guard:lock=_wave_locks.setdefault(mid,threading.Lock())
 with lock:return _waveform(mid,bins)
def _waveform(mid,bins=4096):
 from app import media,DATA
 m=media(mid)
 if m['type']=='image':return {'has_audio':False,'peaks':[],'duration':0}
 cache=DATA/(mid+'.waveform.json');stamp=[m['path'].stat().st_size,m['path'].stat().st_mtime_ns,2,bins]
 if cache.exists():
  try:
   result=json.loads(cache.read_text());
   if result.get('stamp')==stamp:return result
  except (ValueError,OSError):pass
 from audio_spectrum import AudioSpectrum
 spectrum=AudioSpectrum(m['duration'],bins);offset=0
 cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(m['path']),'-vn','-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1']
 # Log to disk so stderr cannot fill while stdout is consumed incrementally.
 logpath=DATA/(mid+'.waveform.log');began=time.monotonic()
 with logpath.open('wb') as log:
  process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  try:
   while True:
    block=process.stdout.read(16384)
    if not block:break
    values=np.frombuffer(block,dtype='<f4');spectrum.add(values);offset+=len(values)
    if time.monotonic()-began>120:raise ValueError('Calcul de la forme d’onde trop long.')
   code=process.wait(timeout=10)
  finally:
   if process.poll() is None:process.kill();process.wait()
 if code and offset==0:
  err=logpath.read_text(errors='replace')
  if 'matches no streams' not in err and 'does not contain any stream' not in err:raise ValueError('Impossible de lire la piste audio.')
 result={'stamp':stamp,'has_audio':offset>0,'duration':m['duration'],**(spectrum.result() if offset else {'peaks':[],'rms':[],'spectrum':[]})}
 temp=cache.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(cache);return result
@router.get('/defaults')
def defaults():
 from app import Config
 return Config(faces=False,heads=False,classes=[],scope='tous').model_dump()
@router.get('/{mid}/waveform')
def get_waveform(mid:str):
 try:return waveform(mid)
 except ValueError as exc:raise HTTPException(422,str(exc))

def run_ffmpeg(cmd,job):
 from app import DATA
 with (DATA/(job['id']+'.montage-ffmpeg.log')).open('ab') as log:
  process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error',*cmd],stdout=subprocess.DEVNULL,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  try:
   while process.poll() is None:
    if job.get('cancel'):raise InterruptedError('Montage annulé.')
    time.sleep(.05)
   if process.returncode:raise ValueError('Encodage du montage impossible. Consultez le journal du montage.')
  finally:
   if process.poll() is None:process.kill();process.wait()

def overlay_png(layer,kind,width,height,path):
 from PIL import Image,ImageDraw,ImageFont
 from engine import rgb_color
 from app import media
 canvas=Image.new('RGBA',(width,height),(0,0,0,0));x,y=round(layer.x*width/100),round(layer.y*height/100)
 if kind=='image':
  with Image.open(media(layer.media_id)['path']) as source:tile=source.convert('RGBA')
  w=max(1,round(width*layer.width/100));h=max(1,round(height*layer.height/100) if layer.height is not None else round(tile.height*w/tile.width));tile=tile.resize((w,h),Image.Resampling.BILINEAR)
  tile.putalpha(tile.getchannel('A').point(lambda a:round(a*layer.opacity/100)));canvas.alpha_composite(tile,(x,y))
 else:
  font=ImageFont.truetype(str(Path(__file__).resolve().parents[1]/'frontend/public/fonts/SFPRODISPLAYREGULAR.OTF'),layer.size);alpha=round(layer.opacity*2.55);color=(*rgb_color(layer.color),alpha)
  ImageDraw.Draw(canvas).text((x,y),layer.text,font=font,fill=color,stroke_width=1,stroke_fill=(0,0,0,alpha//2) if layer.color=='white' else (255,255,255,alpha//2))
 canvas.save(path)

def export_sequence(raw,job):
 from app import media,Config,DATA,export
 from inspection import candidate_seeds
 seq=Sequence.model_validate(raw).validate_media();assets={a.media_id:a for a in seq.assets};visual=sorted([c for c in seq.clips if c.track=='video'],key=lambda c:c.start);audio=[c for c in seq.clips if c.track=='audio' and not c.muted and c.volume]
 total=max([c.start+c.source_out-c.source_in for c in seq.clips]+[z.end for z in [*seq.texts,*seq.images]]);base=media(seq.assets[0].media_id);w=base['width']+(base['width']%2);h=base['height']+(base['height']%2);fps=max(.001,base['fps'] if base['type']=='video' else 25);folder=DATA/(job['id']+'.montage');folder.mkdir(exist_ok=True);generated=[];rendered={};completed=False;output=DATA/(uuid.uuid4().hex+'.mp4')
 job['montage_final']=str(output)
 try:
  for n,c in enumerate(visual):
   mid=c.media_id;asset=assets[mid];cfg=Config.model_validate({**asset.settings,'frame_step':seq.frame_step,'library_only':True})
   if cfg.library_only:cfg=__import__('inspection').library_config(cfg)
   targets=any(not seed.hidden for seed in cfg.custom) or (cfg.discovery_mode and bool(cfg.selected_ids)) or (not cfg.discovery_mode and cfg.scope=='tous' and bool(cfg.faces or cfg.heads or cfg.classes or cfg.plates or cfg.extended or cfg.ocr or cfg.brands or cfg.grounding_prompts))
   visual_changes=cfg.brightness!=100 or cfg.text_overlays or cfg.image_overlays or (cfg.watermark.enabled and cfg.watermark.text)
   if not targets and not visual_changes:
    rendered[c.id]=(media(mid)['path'],False);continue
   if cfg.discovery_mode and not cfg.library_only:cfg.custom += [s for s in candidate_seeds(mid,cfg) if s.id not in {x.id for x in cfg.custom}];cfg.scope='selection'
   cfg=cfg.model_copy(update={'mute':True});m=media(mid);first=min(m['frames']-1,max(0,round(c.source_in*m['fps'])));span=(first,min(m['frames'],max(first+1,round(c.source_out*m['fps'])))) if m['type']=='video' else None
   result=export(mid,cfg,job,frame_range=span)
   rendered[c.id]=(result,True);generated.append(result);job['montage_outputs']=[str(p) for p in generated];job['phase']=f'Passage masqué : {asset.name}';job['progress']=round(35*(n+1)/max(1,len(visual)))
  segments=[];cursor=0
  def black(duration):
   p=folder/f'{len(segments):04d}.mp4';run_ffmpeg(['-f','lavfi','-i',f'color=c=black:s={w}x{h}:r={fps}:d={duration}','-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',str(p)],job);segments.append(p)
  for n,c in enumerate(visual):
   if c.start>cursor+.001:black(c.start-cursor)
   m=media(c.media_id);duration=c.source_out-c.source_in;p=folder/f'{len(segments):04d}.mp4';rendered_path,is_trimmed=rendered[c.id];input_args=['-loop','1'] if m['type']=='image' else ['-ss',str(0 if is_trimmed else c.source_in)]
   job['phase']=f'Assemblage vidéo · clip {n+1}/{len(visual)}';run_ffmpeg([*input_args,'-i',str(rendered_path),'-frames:v',str(max(1,round(duration*fps))),'-an','-vf',f'setpts=PTS-STARTPTS,scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps}:eof_action=pass,tpad=stop_mode=clone:stop_duration={2/fps},setsar=1','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',str(p)],job);segments.append(p);cursor=c.start+duration;job['progress']=35+round(30*(n+1)/max(1,len(visual)))
  if cursor<total-.001:black(total-cursor)
  listing=folder/'segments.txt';listing.write_text(''.join("file '"+p.as_posix()+"'\n" for p in segments),encoding='utf-8')
  inputs=['-f','concat','-safe','0','-i',str(listing)];filters=[];labels=[];groups={}
  for c in audio:
   cfg=Config.model_validate(assets[c.media_id].settings)
   if (not c.source_audio_override and (cfg.mute or not cfg.volume)) or not waveform(c.media_id)['has_audio']:continue
   groups.setdefault(c.media_id,[]).append(c)
  for input_index,(mid,clips) in enumerate(groups.items(),1):
   inputs+=['-i',str(media(mid)['path'])];cfg=Config.model_validate(assets[mid].settings)
   branches=[f's{input_index}_{j}' for j in range(len(clips))]
   filters.append(f'[{input_index}:a:0]'+(f'asplit={len(clips)}' if len(clips)>1 else 'anull')+''.join(f'[{tag}]' for tag in branches))
   for j,c in enumerate(clips):
    label=f'a{input_index}_{j}';filters.append(f'[{branches[j]}]atrim=start={c.source_in}:end={c.source_out},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,volume={c.volume/100*(1 if c.source_audio_override else cfg.volume/100)},adelay={round(c.start*1000)}:all=1[{label}]');labels.append(f'[{label}]')
  video_label='0:v:0';layers=[*[(z,'image') for z in seq.images],*[(z,'text') for z in seq.texts]]
  for n,(layer,kind) in enumerate(layers):
   png=folder/f'overlay-{n}.png';overlay_png(layer,kind,w,h,png);index=1+len(groups)+n;inputs+=['-loop','1','-framerate',str(fps),'-i',str(png)]
   next_label=f'overlay{n}';filters.append(f"[{video_label}][{index}:v:0]overlay=0:0:enable='gte(t,{layer.start})*lt(t,{layer.end})':eof_action=pass[{next_label}]");video_label=next_label
  args=[*inputs]
  if labels:filters.append(''.join(labels)+f'amix=inputs={len(labels)}:normalize=0:duration=longest,alimiter=limit=0.95:level=false:latency=true,apad[outa]')
  if filters:
   filterfile=folder/'composition-filters.txt';filterfile.write_text(';'.join(filters),encoding='utf-8');args+=['-filter_complex_threads','1','-filter_complex_script',str(filterfile)]
  args+=['-map',f'[{video_label}]' if layers else '0:v:0']
  if labels:args+=['-map','[outa]','-c:a','aac','-b:a','192k']
  else:args+=['-an']
  video_args=['-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p'] if layers else ['-c:v','copy']
  job['phase']='Composition des textes et images · mixage audio' if layers else 'Mixage audio et finalisation du montage';job['progress']=85
  run_ffmpeg([*args,'-t',str(total),*video_args,'-movflags','+faststart',str(output)],job);job['progress']=100;job['output']=str(output);completed=True;return output
 finally:
  # Every target is a known generated file inside DATA; never delete an imported source.
  for p in generated:
   if p.parent.resolve()==DATA.resolve() and p.name not in {media(a.media_id)['path'].name for a in seq.assets}:p.unlink(missing_ok=True)
  if folder.resolve().parent==DATA.resolve():shutil.rmtree(folder,ignore_errors=True)
  if not completed:output.unlink(missing_ok=True)
