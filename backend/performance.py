"""Per-worker local timings. Wall clocks, actual session providers and bounded sampling."""
import time,json,os,threading,functools,contextlib
from pathlib import Path
from collections import OrderedDict
_current=None
_installed=False
_sessions=OrderedDict()
class Performance:
 def __init__(self,job,folder):
  self.job=job;self.start=time.perf_counter();self.stats={};self.models={};self.lock=threading.RLock();self.stage='Chargement des modèles';self.done=0;self.total=0;self.unit='images';self.phase_start=self.start;self.stop=threading.Event();self.last=0.;self.peak_ram=0.;self.peak_cpu=0.;self.gpu=None;self.resources={};self.progress=0;self.publish_lock=threading.RLock();self.log_last=0.;self.video_done=0;self.video_total=0;self.complete=False
  self.path=Path(folder)/(str(job.get('id','analysis'))+'.performance.jsonl');self.summary=self.path.with_suffix('.json')
 def begin(self):
  global _current
  _current=self;install();self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();self.publish(force=True);return self
 def record(self,name,seconds):
  with self.lock:
   row=self.stats.setdefault(name,{'calls':0,'seconds':0.});row['calls']+=1;row['seconds']+=seconds
 def phase(self,name,total=0,unit='images'):
  self.record('stage/'+self.stage,time.perf_counter()-self.phase_start)
  self.stage=name;self.done=0;self.total=total;self.unit=unit;self.phase_start=time.perf_counter();self.job['phase']=name;self.publish(force=True)
 def tick(self,done,total=None):
  self.done=done
  if self.stage=='Détection':self.video_done=done;self.video_total=total or self.total
  if total is not None:self.total=total
  self.publish(force=bool(self.total and done>=self.total))
 def publish(self,force=False):
  with self.publish_lock:self._publish(force)
 def _publish(self,force=False):
  now=time.perf_counter()
  if not force and now-self.last<.5:return
  self.last=now;elapsed=now-self.phase_start;rate=self.done/elapsed if elapsed and self.done else 0
  bounds={'Chargement des modèles':(0,5),'Détection':(5,55),'Détection approfondie':(55,75),'Association des sujets':(73,75),'Tracking':(75,85),'Vérification temporelle':(85,97),'Recherche par apparence':(97,98),'Protection des apparitions':(97,98),'Préparation des contours':(75,95),'Finalisation':(97,99)};a,b=(0,75) if self.stage=='Protection des apparitions' and self.job.get('kind')=='analyze' else bounds.get(self.stage,(0,99));value=a+(b-a)*min(1,self.done/self.total) if self.total else a;self.progress=max(self.progress,value)
  with self.lock:
   metrics={k:{'calls':v['calls'],'seconds':round(v['seconds'],4),'mean_ms':round(v['seconds']/max(1,v['calls'])*1000,3)} for k,v in self.stats.items()};models={k:dict(v) for k,v in list(self.models.items())}
  state={'tracking_reused':bool(self.job.get('reused_analysis')),'stage':self.stage,'processed':self.done,'total':self.total,'unit':self.unit,'fps':round(rate,2) if self.unit=='images' else None,'units_per_second':round(rate,2),'elapsed_seconds':round(now-self.start,2),'eta_seconds':round((self.total-self.done)/rate) if rate and self.total>=self.done else None,'eta_scope':'étape','video_frames_processed':self.video_done,'video_frames_total':self.video_total,'overall_fps':round(self.video_done/max(.001,now-self.start),2),'resources':self.resources,'peak_ram_mb':round(self.peak_ram,1),'peak_cpu_percent':round(self.peak_cpu,1),'models':models,'timings':metrics}
  self.job['performance']=state
  # Montage has its own multi-clip progress; sampling must never overwrite it.
  if self.complete or self.job.get('state') in {'done','review'}:self.job['progress']=100
  elif self.job.get('kind') not in {'montage','export'}:self.job['progress']=round(self.progress)
  try:
   if not force and now-self.log_last<5:return
   self.log_last=now
   with self.path.open('a',encoding='utf-8') as f:f.write(json.dumps(state,ensure_ascii=False)+'\n')
  except OSError:pass
 def sample(self):
  import psutil
  proc=psutil.Process();proc.cpu_percent();gpu=GpuCounter()
  while not self.stop.wait(.5):
   try:
    ram=proc.memory_info().rss/1024**2;cpu=proc.cpu_percent()/max(1,psutil.cpu_count());memory=psutil.virtual_memory();io=proc.io_counters();self.peak_ram=max(self.peak_ram,ram);self.peak_cpu=max(self.peak_cpu,cpu);self.resources={'cpu_percent':round(cpu,1),'ram_mb':round(ram,1),'available_ram_mb':round(memory.available/1024**2),'io_read_mb':round(io.read_bytes/1024**2,1),'gpu_engine_percent':gpu.read(),'gpu_metric':'busiest process GPU engine; null if unavailable'};self.publish()
   except (psutil.Error,OSError):pass
  gpu.close()
 def finish(self):
  global _current
  self.record('stage/'+self.stage,time.perf_counter()-self.phase_start)
  self.stop.set();self.thread.join(2);self.publish(force=True)
  try:self.summary.write_text(json.dumps(self.job.get('performance',{}),ensure_ascii=False,indent=2),encoding='utf-8')
  except OSError:pass
  _sessions.clear();__import__('video_io').clear_analysis_cache();_current=None

class GpuCounter:
 def __init__(self):
  self.query=None
  try:
   import ctypes as c
   from ctypes import wintypes as w
   self.c=c;self.w=w;self.p=c.WinDLL('pdh');self.query=c.c_void_p();self.counter=c.c_void_p()
   self.p.PdhOpenQueryW.argtypes=[w.LPCWSTR,c.c_size_t,c.POINTER(c.c_void_p)];self.p.PdhAddEnglishCounterW.argtypes=[c.c_void_p,w.LPCWSTR,c.c_size_t,c.POINTER(c.c_void_p)];self.p.PdhCollectQueryData.argtypes=[c.c_void_p];self.p.PdhGetFormattedCounterArrayW.argtypes=[c.c_void_p,w.DWORD,c.POINTER(w.DWORD),c.POINTER(w.DWORD),c.c_void_p];self.p.PdhCloseQuery.argtypes=[c.c_void_p]
   if self.p.PdhOpenQueryW(None,0,c.byref(self.query)) or self.p.PdhAddEnglishCounterW(self.query,'\\GPU Engine(*)\\Utilization Percentage',0,c.byref(self.counter)):self.close();return
   self.p.PdhCollectQueryData(self.query)
  except (OSError,AttributeError):self.query=None
 def read(self,system=False):
  if not self.query:return None
  c=self.c;w=self.w
  class Value(c.Structure):_fields_=[('status',w.DWORD),('value',c.c_double)]
  class Item(c.Structure):_fields_=[('name',w.LPWSTR),('data',Value)]
  self.p.PdhCollectQueryData(self.query);size=w.DWORD();count=w.DWORD();self.p.PdhGetFormattedCounterArrayW(self.counter,0x200,c.byref(size),c.byref(count),None)
  if not size.value:return None
  buf=c.create_string_buffer(size.value)
  if self.p.PdhGetFormattedCounterArrayW(self.counter,0x200,c.byref(size),c.byref(count),buf):return None
  items=c.cast(buf,c.POINTER(Item));engines={};values=[]
  for i in range(count.value):
   name=items[i].name or '';value=items[i].data.value
   if items[i].data.status not in {0,1}:continue
   if system:
    # Sum processes sharing the same physical engine, then report busiest engine.
    key=__import__('re').sub(r'^pid_\d+_','',name);engines[key]=engines.get(key,0)+max(0,value)
   elif f'pid_{os.getpid()}_' in name:values.append(value)
  if system:values=list(engines.values())
  return round(min(100,max(values)),1) if values else (None if system else 0.)
 def close(self):
  if self.query:self.p.PdhCloseQuery(self.query);self.query=None

def phase(name,total=0,unit='images'):
 if _current:_current.phase(name,total,unit)
def tick(done,total=None):
 if _current:_current.tick(done,total)
@contextlib.contextmanager
def timed(name):
 start=time.perf_counter()
 try:yield
 finally:
  if _current:_current.record(name,time.perf_counter()-start)
def profiled(fn):
 @functools.wraps(fn)
 def run(mid,cfg,job,*a,**k):
  owned=_current is None;p=Performance(job,__import__('app').DATA).begin() if owned else _current
  try:
   result=fn(mid,cfg,job,*a,**k)
   if owned:p.complete=True
   return result
  finally:
   if owned:p.finish()
 return run

def install():
 global _installed
 if _installed:return
 _installed=True
 import onnxruntime as ort
 Original=ort.InferenceSession
 class Session(Original):
  def __new__(cls,path,*a,**k):
   providers=k.get('providers');key=(str(path),str(providers),getattr(a[0] if a else k.get('sess_options'),'logid',''))
   if _current and key in _sessions:
    obj=_sessions[key];_sessions.move_to_end(key);return obj
   obj=super().__new__(cls);obj._metric_key=key;return obj
  def __init__(self,path,*a,**k):
   if getattr(self,'_measured_ready',False):return
   start=time.perf_counter();opts=a[0] if a else k.get('sess_options')
   if opts is None:
    opts=ort.SessionOptions()
    if a:a=(opts,*a[1:])
    else:k['sess_options']=opts
   opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;opts.enable_mem_pattern=False;opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;opts.add_session_config_entry('session.intra_op.allow_spinning','0');opts.log_severity_level=3
   super().__init__(path,*a,**k);self._name=Path(path).name;self._measured_ready=True
   if _current:_sessions[self._metric_key]=self
   while len(_sessions)>8:_sessions.popitem(last=False)
   if _current:
    _current.record('load/'+self._name,time.perf_counter()-start);row=_current.models.setdefault(self._name,{'providers':self.get_providers(),'primary':self.get_providers()[0],'sessions':[],'note':'session provider; mixed CPU operators remain possible'});row['sessions'].append(self.get_providers()[0])
  def run(self,*a,**k):
   with timed('inference/'+self._name):
    try:
     result=super().run(*a,**k)
     if _current:
      _current.models.setdefault(self._name,{'providers':self.get_providers(),'primary':self.get_providers()[0]});shape='x'.join(map(str,next(iter(a[1].values())).shape)) if len(a)>1 else 'unknown';_current.models[self._name].setdefault('executed_shapes',{})[shape]=self.get_providers()[0]
     return result
    except Exception as exc:
     if _current:_current.models[self._name]['last_error']=str(exc)[:300]
     _sessions.pop(self._metric_key,None);raise
 ort.InferenceSession=Session
 # Total stage costs supplement per-ONNX timings; nested timings must not be added together.
 for module,cls,method,label in [('videomask.faces','YuNetDetector','__init__','load/YuNet'),('videomask.faces','YuNetDetector','detect','inference/YuNet'),('videomask.custom_tracking','ObjectFollower','update','tracking/CSRT'),('engine','MultiTracker','update','tracking/ByteTrack'),('advanced','AppearanceTracker','update','tracking/BoT-SORT'),('advanced','Segmenter','__init__','load/SAM2'),('advanced','Segmenter','annotate','inference/SAM2'),('advanced','GroundedDetector','__init__','load/GroundingDINO'),('advanced','GroundedDetector','detect','inference/GroundingDINO')]:
  target=getattr(__import__(module,fromlist=[cls]),cls);original=getattr(target,method)
  def wrap(fn,name):
   @functools.wraps(fn)
   def measured(*a,**k):
    actual_name=('tracking/OC-SORT' if getattr(a[0],'algorithm',None)=='ocsort' else name) if name=='tracking/ByteTrack' else name
    with timed(actual_name):
     result=fn(*a,**k)
     if _current and name in {'inference/YuNet','load/YuNet','load/SAM2','load/GroundingDINO'}:_current.models.setdefault(name.split('/',1)[1],{'providers':['OpenCV CPU' if 'YuNet' in name else 'PyTorch CPU'],'primary':'CPUExecutionProvider'})
     return result
   return measured
  setattr(target,method,wrap(original,label))

 from sparse_motion import SparseMotion
 for method,label in [('infer','tracking/shared_foreground_flow'),('commit','tracking/foreground_points')]:
  setattr(SparseMotion,method,wrap(getattr(SparseMotion,method),label))

 # Decoder time counts actual decoding, excluding frame-cache hits.
 import video_io
 original_decode=video_io._decode
 @functools.wraps(original_decode)
 def measured_decode(*a,**k):
  with timed('decode/video'):return original_decode(*a,**k)
 video_io._decode=measured_decode

def note_model(name,message,provider=None):
 if _current:
  row=_current.models.setdefault(name,{});notes=row.setdefault('notes',[])
  if provider:row.update(primary=provider,providers=[provider])
  if message not in notes:notes.append(message)
