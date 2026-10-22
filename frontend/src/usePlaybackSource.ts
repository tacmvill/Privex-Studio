import {useCallback,useEffect,useRef,useState,type RefObject} from 'react';
const prepared=new Map<string,string>();
export function usePlaybackSource(mid?:string,original=''){
 const [url,setUrl]=useState(original),[fallback,setFallback]=useState(false),[message,setMessage]=useState('');
 useEffect(()=>{setUrl(mid&&prepared.get(mid)||original);setFallback(false);setMessage('')},[mid,original]);
 const failed=useCallback((error?:unknown)=>{
  const name=(error as {name?:string})?.name;
  if(name==='AbortError'||name==='NotAllowedError')return;
  if(mid&&!prepared.has(mid))setFallback(true);else setMessage('Lecture indisponible pour ce fichier.');
 },[mid]);
 useEffect(()=>{
  if(!fallback||!mid)return;let disposed=false,timer=0;const controller=new AbortController();
  const poll=async()=>{try{
   const r=await fetch(`/api/media/${mid}/playback`,{method:'POST',signal:controller.signal});const data=await r.json();if(disposed)return;
   if(!r.ok||data.state==='error'){setMessage(data.message||'Préparation de lecture impossible.');return}
   if(data.state==='ready'){prepared.set(mid,data.url);setUrl(data.url);setMessage('');setFallback(false)}else{setMessage(data.message);timer=window.setTimeout(poll,500)}
  }catch(e){if(!disposed&&(e as Error).name!=='AbortError')setMessage('Le moteur de lecture est indisponible.')}};
  void poll();return()=>{disposed=true;controller.abort();window.clearTimeout(timer)};
 },[fallback,mid]);
 return {url,failed,message};
}

/** Transport state changes start playback; clock ticks only correct large drift. */
export function useVideoTransport(ref:RefObject<HTMLMediaElement|null>,url:string,at:number,playing:boolean,onFailure:(e?:unknown)=>void,identity=''){
 const fail=useRef(onFailure);fail.current=onFailure;const position=useRef(at);position.current=at;
 useEffect(()=>{
  const v=ref.current;if(!v)return;
  const play=()=>{if(playing)void v.play().catch(e=>{if(e.name!=='AbortError'&&e.name!=='NotAllowedError')fail.current(e)});else v.pause()};
  const ready=()=>{if(Math.abs(v.currentTime-position.current)>.1)v.currentTime=position.current;play()};
  v.addEventListener('loadedmetadata',ready);v.addEventListener('canplay',play);play();
  return()=>{v.removeEventListener('loadedmetadata',ready);v.removeEventListener('canplay',play)};
 },[url,playing,identity]);
 useEffect(()=>{const v=ref.current;if(v&&v.readyState>=1&&Math.abs(v.currentTime-at)>(playing ? .6 : .01))v.currentTime=at},[at,url,playing]);
}
