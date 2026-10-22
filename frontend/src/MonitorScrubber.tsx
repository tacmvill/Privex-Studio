import {frameTimecode} from './timecode';
import {useRef,useState} from 'react';
export function MonitorScrubber({time,duration,fps,disabled,onSeek,previewAt}:{time:number;duration:number;fps:number;disabled:boolean;onSeek:(t:number)=>void;previewAt:(t:number)=>string|null}){
 const host=useRef<HTMLDivElement>(null),[hover,setHover]=useState<{time:number;x:number}|null>(null);
 const at=hover?Math.min(Math.max(0,duration-1/fps),Math.round(hover.time*4)/4):0,src=hover?previewAt(at):null;
 return <div ref={host} className="monitor-scrubber" onPointerLeave={()=>setHover(null)}><input aria-label="Position dans la vidéo du moniteur" type="range" min={0} max={Math.max(0,duration-1/fps)} step={1/fps} value={Math.min(time,Math.max(0,duration-1/fps))} disabled={disabled} onChange={e=>onSeek(+e.target.value)} onPointerMove={e=>{const rect=e.currentTarget.getBoundingClientRect();setHover({time:Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width))*duration,x:Math.max(82,Math.min(rect.width-82,e.clientX-rect.left))})}} style={{'--scrub-progress':`${duration?Math.min(100,time/duration*100):0}%`} as React.CSSProperties}/>{hover&&<div className="monitor-scrub-preview" style={{left:hover.x}}>{src?<img src={src} alt="Aperçu de la position dans la vidéo"/>:<span>Aucun média</span>}<code>{frameTimecode(at,fps)}</code></div>}</div>;
}
