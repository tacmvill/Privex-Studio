type Clip={track:string;media_id:string;source_in:number;source_out:number};
export function detectionScope(clips:Clip[]|undefined,media:{id:string;frames:number;fps:number},samples:number[]){
 const raw=(clips===undefined?[{track:'video',media_id:media.id,source_in:0,source_out:media.frames/media.fps}]:clips).filter(c=>c.track==='video'&&c.media_id===media.id).map(c=>[Math.max(0,Math.round(c.source_in*media.fps)),Math.min(media.frames,Math.round(c.source_out*media.fps))]).filter(([a,b])=>b>a).sort((a,b)=>a[0]-b[0]);const ranges:number[][]=[];
 for(const [a,b] of raw){const previous=ranges.at(-1);if(previous&&a<=previous[1])previous[1]=Math.max(previous[1],b);else ranges.push([a,b])}
 if(!ranges.length)return {ranges,times:[]};
 const frame=Math.round((samples[0]||0)*media.fps);
 const nearest=ranges.reduce((best,[a,b])=>{const f=Math.max(a,Math.min(b-1,frame));return Math.abs(f-frame)<Math.abs(best-frame)?f:best},ranges[0][0]);
 return {ranges,times:[nearest/media.fps]};
}
