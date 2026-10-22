import type {TextLayer,ImageLayer} from './LayerEditor';
import type {ProjectData} from './useProjects';
export type SequenceMedia={id:string;type:'image'|'video';width:number;height:number;fps:number;duration:number;frames:number;name?:string};
export type SequenceAsset={media_id:string;name:string;settings:Record<string,unknown>;project:ProjectData & {media:SequenceMedia}};
export type TimelineClip={id:string;media_id:string;track:'video'|'audio';start:number;source_in:number;source_out:number;volume:number;muted:boolean;source_audio_override?:boolean;link?:string|null};
export type TimedText=TextLayer&{start:number;end:number};
export type TimedImage=ImageLayer&{start:number;end:number};
export type SequenceDoc={assets:SequenceAsset[];clips:TimelineClip[];texts?:TimedText[];images?:TimedImage[]};
export const uid=()=>crypto.randomUUID();
export const length=(c:TimelineClip)=>c.source_out-c.source_in;
/** Match the export frame rate, based on the first asset in the montage. */
export function sequenceFrameRate(d:SequenceDoc|null|undefined):number{const m=d?.assets[0]?.project.media;return m?.type==='video'&&Number.isFinite(m.fps)&&m.fps>0?Math.max(.001,m.fps):25}
export const sequenceDuration=(d:SequenceDoc)=>Math.max(0,...d.clips.map(c=>c.start+length(c)),...(d.texts||[]).map(c=>c.end),...(d.images||[]).map(c=>c.end));
export function appendAsset(d:SequenceDoc,asset:SequenceAsset):SequenceDoc{const first=d.assets[0]?.project.media||asset.project.media,fps=first.type==='video'?Math.max(.001,first.fps||25):25;const duration=asset.project.media.type==='image'?Math.round(5*fps)/fps:Math.max(1/fps,Math.floor(asset.project.media.duration*fps+1e-6)/fps);const start=Math.ceil(sequenceDuration(d)*fps-1e-6)/fps,link=uid(),clip={media_id:asset.media_id,start,source_in:0,source_out:duration,volume:100,muted:false,link};return {...d,assets:d.assets.some(a=>a.media_id===asset.media_id)?d.assets:[...d.assets,asset],clips:[...d.clips,{...clip,id:uid(),track:'video'},...(asset.project.media.type==='video'?[{...clip,id:uid(),track:'audio' as const}]:[])]}}
/** Ripple visual edits; attached audio moves by the same delta, independent audio stays put. */
function ripple(clips:TimelineClip[]):TimelineClip[]{let cursor=0;const deltas=new Map<string,number>();const visual=clips.filter(c=>c.track==='video').map(c=>{const delta=cursor-c.start;if(c.link)deltas.set(c.link,delta);const next={...c,start:cursor};cursor+=length(c);return next});return [...visual,...clips.filter(c=>c.track==='audio').map(c=>({...c,start:Math.max(0,c.start+(c.link?deltas.get(c.link)||0:0))}))]}
export function splitAt(d:SequenceDoc,time:number,mode:'both'|'video'|'audio',fps:number):SequenceDoc{const step=1/fps,t=Math.round(time*fps)/fps;const targets=d.clips.filter(c=>(mode==='both'||mode===c.track)&&t-c.start>=step-.0001&&c.start+length(c)-t>=step-.0001);if(!targets.length)return d;const unlink=new Set(mode==='both'?[]:targets.map(c=>c.link));const rightLinks=new Map<string,string>();return {...d,clips:d.clips.flatMap(c=>{const base={...c,link:c.link&&unlink.has(c.link)?null:c.link};if(!targets.includes(c))return [base];if(base.link&&!rightLinks.has(base.link))rightLinks.set(base.link,uid());const cut=c.source_in+t-c.start;return [{...base,source_out:cut},{...base,id:uid(),start:t,source_in:cut,link:base.link?rightLinks.get(base.link):null}]})}}
export function removeClip(d:SequenceDoc,id:string,linked:boolean):SequenceDoc{const c=d.clips.find(c=>c.id===id);if(!c)return d;return {...d,clips:ripple(d.clips.filter(z=>z.id!==id&&!(linked&&c.link&&z.link===c.link)).map(z=>!linked&&c.link&&z.link===c.link?{...z,link:null}:z))}}
/** Non-ripple trim: changing the in point moves only the left edge; the out point stays fixed. */
export function updateClip(d:SequenceDoc,id:string,patch:Partial<TimelineClip>,linked:boolean):SequenceDoc{
 const c=d.clips.find(c=>c.id===id);if(!c)return d;const timing=patch.source_in!==undefined||patch.source_out!==undefined;
 const apply=(z:TimelineClip,value:Partial<TimelineClip>)=>{const next={...z,...value};if(value.source_in!==undefined&&value.start===undefined){next.source_in=Math.max(0,z.source_in-z.start,value.source_in);next.start=z.start+next.source_in-z.source_in}return next};
 return {...d,clips:d.clips.map(z=>{if(z.id===id)return apply(z,{...patch,...(!linked&&timing?{link:null}:{})});if(patch.link===null&&c.link&&z.link===c.link)return {...z,link:null};if(linked&&timing&&c.link&&z.link===c.link)return apply(z,{...(patch.source_in!==undefined?{source_in:patch.source_in}:{}),...(patch.source_out!==undefined?{source_out:patch.source_out}:{})});return !linked&&timing&&c.link&&z.link===c.link?{...z,link:null}:z})};
}
export function moveClip(d:SequenceDoc,id:string,direction:number):SequenceDoc{const visuals=d.clips.filter(c=>c.track==='video').sort((a,b)=>a.start-b.start),i=visuals.findIndex(c=>c.id===id),next=i+direction;if(i<0||next<0||next>=visuals.length)return d;[visuals[i],visuals[next]]=[visuals[next],visuals[i]];return {...d,clips:ripple([...visuals,...d.clips.filter(c=>c.track==='audio')])}}

/** PC mute affects listening only; project/clip gain is what gets exported. */
export function audioState(clip:TimelineClip,asset:SequenceAsset){const raw=Number(clip.volume??100),volume=Number.isFinite(raw)?raw:100,setting=Number(asset.settings.volume??100),source=clip.source_audio_override?1:(Number.isFinite(setting)?setting:100)/100;return {gain:Math.max(0,volume/100*source),muted:!!clip.muted||volume<=0||(!clip.source_audio_override&&(!!asset.settings.mute||source<=0))}}

/** Map a source seek onto a retained clip; never open the uncut source as a fallback. */
export function sourceToSequenceTime(d:SequenceDoc,mediaId:string,at:number):number|null{
 const clips=d.clips.filter(c=>c.track==='video'&&c.media_id===mediaId).sort((a,b)=>a.start-b.start);if(!clips.length)return null;
 const c=clips.find(c=>at>=c.source_in&&at<c.source_out)||clips.reduce((best,c)=>Math.min(Math.abs(at-c.source_in),Math.abs(at-c.source_out))<Math.min(Math.abs(at-best.source_in),Math.abs(at-best.source_out))?c:best);
 const step=1/sequenceFrameRate(d);return c.start+Math.max(0,Math.min(length(c)-step,at-c.source_in));
}
