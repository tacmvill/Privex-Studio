import {useCallback,useEffect,useMemo,useRef,useState,type RefObject} from 'react';
export type TrackedBox={id:string;class:string;color:string;box:number[];category?:string;reference_ids?:string[];visible?:boolean;existence?:string;highlighted?:boolean};
export type PreviewMask={id:string;box:number[];category?:string;class?:string;polygons?:number[][][];privacy_coverage?:boolean;style?:Record<string,unknown>};
export type TrackingSource={passiveClock?:boolean;mid:string;ref:string;time:number;fps:number;frames?:number;endTime?:number;playing:boolean;enabled:boolean;settings?:Record<string,unknown>;video?:RefObject<HTMLVideoElement|null>};
/** Bounded, persistent prefetch. Only identity changes abort in-flight windows. */
export function useTrackedBoxes(source?:TrackingSource){
 const [error,setError]=useState(''),[clock,setClock]=useState(source?.time||0),[generation,setGeneration]=useState(0);
 const cache=useRef(new Map<number,TrackedBox[]>()),masks=useRef(new Map<number,PreviewMask[]>()),loaded=useRef(new Set<number>()),uncovered=useRef(new Set<number>());
 const pending=useRef(new Map<number,AbortController>()),attempts=useRef(new Map<number,number>()),failures=useRef(new Map<number,string>()),timers=useRef(new Map<number,number>()),epoch=useRef(0);
 const settingsKey=useMemo(()=>source?.settings?JSON.stringify(source.settings):'',[source?.settings]),identity=`${source?.mid}:${source?.ref}:${settingsKey}`,cachedIdentity=useRef(identity);
 const retry=()=>{attempts.current.clear();failures.current.clear();setError('');setGeneration(v=>v+1)};
 useEffect(()=>{
  const version=++epoch.current;cachedIdentity.current=identity;setError('');cache.current.clear();masks.current.clear();loaded.current.clear();uncovered.current.clear();attempts.current.clear();failures.current.clear();setGeneration(v=>v+1);
  return()=>{if(epoch.current===version)epoch.current++;for(const c of pending.current.values())c.abort();pending.current.clear();for(const t of timers.current.values())clearTimeout(t);timers.current.clear()};
 },[identity,source?.enabled]);
 useEffect(()=>{if(!source?.playing)setClock(source?.time||0)},[source?.time,source?.playing]);
 useEffect(()=>{
  const video=source?.video?.current;if(!video||!source?.playing||!source.enabled)return;let active=true,handle=0,lastUpdate=-Infinity;
  const step=(_now:number,meta?:VideoFrameCallbackMetadata)=>{if(!active)return;if(!source.passiveClock||_now-lastUpdate>=100){setClock(meta?.mediaTime??video.currentTime);lastUpdate=_now;}handle=video.requestVideoFrameCallback?video.requestVideoFrameCallback(step):requestAnimationFrame(step)};
  handle=video.requestVideoFrameCallback?video.requestVideoFrameCallback(step):requestAnimationFrame(step);
  return()=>{active=false;if(video.cancelVideoFrameCallback)video.cancelVideoFrameCallback(handle);else cancelAnimationFrame(handle)};
 },[source?.playing,source?.enabled,identity]);
 const fps=source?.fps||25,lastFrame=Math.min((source?.frames??Infinity)-1,source?.endTime==null?Infinity:Math.max(0,Math.ceil(source.endTime*fps-1e-6)-1));
 const cursor=source?.playing&&source.video?clock:source?.time||0,at=Math.max(0,Math.min(cursor,lastFrame/fps)),block=Math.floor(at/2),requestedBlock=Math.floor((source?.time||0)/2);
 useEffect(()=>{
  if(!source?.enabled||!source.ref)return;const version=epoch.current;
  for(const b of new Set([requestedBlock,block,block+1,block+2])){
   const end=b*2+2;if(loaded.current.has(b)||pending.current.has(b)||timers.current.has(b))continue;
   const count=attempts.current.get(b)||0;if(count>=3){if(b===block||b===requestedBlock)setError(failures.current.get(b)||'Suivi indisponible. Réessayez.');continue}
   const controller=new AbortController();pending.current.set(b,controller);attempts.current.set(b,count+1);let timedOut=false;
   const timeout=window.setTimeout(()=>{timedOut=true;controller.abort()},8000);
   fetch(`/api/inspection/${source.ref}/tracking-window?mid=${source.mid}&start=${b*2}&end=${end}`,{signal:controller.signal,...(source.settings?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({media_id:source.mid,settings:source.settings})}:{})}).then(async r=>{
    const data=await r.json();if(!r.ok){const e=Object.assign(new Error(typeof data.detail==='string'?data.detail:'Suivi de lecture indisponible.'),{transient:[429,502,503,504].includes(r.status)});throw e}return data;
   }).then((data:{available:boolean;frames:{frame:number;objects:TrackedBox[];masks?:PreviewMask[]|null}[]})=>{
    if(epoch.current!==version)return;if(!data.available)throw Error('Ce suivi n’est pas disponible. Relancez Détecter puis Appliquer le masquage.');
    loaded.current.add(b);failures.current.delete(b);setError('');for(const row of data.frames){cache.current.set(row.frame,row.objects);if(row.masks===null)uncovered.current.add(row.frame);else masks.current.set(row.frame,row.masks||[])}setGeneration(v=>v+1);
   }).catch(e=>{
    if(epoch.current!==version||e.name==='AbortError'&&!timedOut)return;
    const transient=timedOut||e instanceof TypeError||e instanceof SyntaxError||e.transient;
    const message=transient?'Connexion au suivi interrompue. Réessayez.':e.message;failures.current.set(b,message);
    if(transient&&count+1<3){timers.current.set(b,window.setTimeout(()=>{timers.current.delete(b);if(epoch.current===version)setGeneration(v=>v+1)},400*(count+1)))}else{attempts.current.set(b,3);setError(message)}
   }).finally(()=>{clearTimeout(timeout);if(pending.current.get(b)===controller)pending.current.delete(b)});
  }
  for(const frame of cache.current.keys())if(Math.abs(Math.floor(frame/fps/2)-block)>4){cache.current.delete(frame);masks.current.delete(frame);uncovered.current.delete(frame)}
  for(const [b,c] of pending.current)if(Math.abs(b-block)>4){c.abort();pending.current.delete(b);attempts.current.delete(b)}
  for(const b of loaded.current)if(Math.abs(b-block)>4){loaded.current.delete(b);attempts.current.delete(b)}
 },[identity,block,requestedBlock,source?.enabled,source?.endTime,generation]);
 const frame=Math.min(lastFrame,Math.max(0,Math.floor(at*fps+.0001))),readyFrame=Math.min(lastFrame,Math.max(0,Math.floor((source?.time||0)*fps+.0001)));
 const firstFrame=source?.playing?frame:readyFrame,endFrame=Math.min(lastFrame,firstFrame+Math.max(2,Math.ceil(fps*.3)));
 const lookup=useCallback((time:number)=>{const i=Math.max(0,Math.min(lastFrame,Math.floor(time*fps+.0001)));return {boxes:cache.current.get(i),masks:masks.current.get(i)}},[fps,lastFrame,identity,generation]);
 const ready=!!source?.enabled&&cachedIdentity.current===identity&&Array.from({length:Math.max(1,endFrame-firstFrame+1)},(_,i)=>firstFrame+i).every(i=>source?.settings?masks.current.has(i):cache.current.has(i));
 const scopeError=source?.settings&&(uncovered.current.has(readyFrame)||uncovered.current.has(frame))?'Ce passage n’a pas été analysé. Relancez Détecter sur le montage actuel.':'';
 return {lookup,time:at,error:scopeError||((attempts.current.get(block)||0)>=3?failures.current.get(block)||error:(attempts.current.get(requestedBlock)||0)>=3?failures.current.get(requestedBlock)||error:''),ready,retry,masks:source?.enabled&&cachedIdentity.current===identity?masks.current.get(frame):undefined,boxes:source?.enabled&&cachedIdentity.current===identity?cache.current.get(frame):undefined};
}
