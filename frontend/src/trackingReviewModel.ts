import type {TemporalDecision} from './temporalModel';
import type {SequenceAsset,TimelineClip} from './sequence';
export type TrackingGap={id:string;subject_id:string;label:string;start:number;end:number;reason:string;kind?:'detection'|'tracking'|'identity'|'edit';focus_time?:number;box?:number[];detection_confidence?:number;tracking_confidence?:number;media_id:string;media_name?:string;clip_id?:string;timeline_start?:number;timeline_end?:number;reference_time?:number;reference_box?:number[];reference_ids?:string[];color?:string;category?:string};
export type TrackingDecision={kind:'anchor';gap:TrackingGap;time:number;box:number[]}|{kind:'absent';gap:TrackingGap}|{kind:'temporal';gap:TrackingGap;action:TemporalDecision['action'];scope:'frame'|'interval'|'trajectory';time:number;target_id?:string;polygon?:number[][];box?:number[]}|{kind:'contour';gap:TrackingGap;time:number;box:number[];polygon:number[][];scope:'frame'|'interval'|'trajectory'};
type Reference={id:string;time:number;box:number[];label:string;color:string;category?:string;hidden?:boolean;anchors?:{time:number;box:number[]}[];style?:unknown};
export const subjectKey=(id:string)=>id.startsWith('m-da')?id.slice(3):id;
export const reviewKey=(g:TrackingGap)=>`${g.media_id}:${g.clip_id||''}:${g.id}`;
export function gapForClip(gap:TrackingGap,clip:TimelineClip,asset:SequenceAsset){
 const start=Math.max(gap.start,clip.source_in),end=Math.min(gap.end,clip.source_out),fps=asset.project.media.fps||25;
 return {...gap,...(start!==gap.start||end!==gap.end?{id:`${gap.subject_id}:${Math.round(start*fps)}:${Math.round(end*fps)-1}:${gap.reason}`} : {}),start,end,clip_id:clip.id,timeline_start:clip.start+start-clip.source_in,timeline_end:clip.start+end-clip.source_in,positionStart:clip.start+start-clip.source_in,positionEnd:clip.start+end-clip.source_in};
}
export function confirmedGap(gap:TrackingGap,confirmed:string[],fps:number){
 if(confirmed.includes(gap.id))return true;const aliases=new Set([subjectKey(gap.subject_id),...(gap.reference_ids||[]).map(subjectKey)]),spans:number[][]=[];
 for(const value of confirmed){const parts=value.split(':'),end=Number(parts.at(-2)),start=Number(parts.at(-3)),subject=parts.slice(0,-3).join(':');if(parts.length>=4&&Number.isInteger(start)&&Number.isInteger(end)&&start<=end&&aliases.has(subjectKey(subject)))spans.push([start,end])}
 let cursor=Math.round(gap.start*fps);const end=Math.round(gap.end*fps)-1;
 for(const [a,b] of spans.sort((a,b)=>a[0]-b[0])){if(b<cursor)continue;if(a>cursor)return false;cursor=Math.max(cursor,b+1);if(cursor>end)return true}return false;
}
export function reviewSubject(asset:SequenceAsset,gap:TrackingGap){
 const key=subjectKey(gap.subject_id),cfg=asset.settings;
 const seed=([...(cfg.custom as Reference[]||[]),...(cfg.references as Reference[]||[])]).find(s=>subjectKey(s.id)===key);
 const ref=seed||((asset.project.ui.candidates as {id:string;time:number;box:number[];class:string;color:string;category?:string}[]||[]).filter(c=>subjectKey(c.id)===key).map(c=>({...c,label:c.class}))[0]);
 const override=(cfg.subject_overrides as Record<string,{class?:string;color?:string}>||{})[key];
 const number=(asset.project.ui.numbers as [string,number][]||[]).find(([id])=>subjectKey(id)===key)?.[1];
 return {id:gap.subject_id,label:override?.class||ref?.label||gap.label,color:override?.color||ref?.color||gap.color||'#0a84ff',number,referenceTime:ref?.time??gap.reference_time,referenceBox:ref?.box??gap.reference_box,seed:ref};
}
/** A decision changes this one subject; all other subjects and their styles remain intact. */
export function applyTrackingDecision(asset:SequenceAsset,decision:TrackingDecision):SequenceAsset{
 if(decision.kind==='contour'){const anchored=applyTrackingDecision(asset,{kind:'anchor',gap:decision.gap,time:decision.time,box:decision.box});return applyTrackingDecision(anchored,{kind:'temporal',action:'contour',gap:decision.gap,time:decision.time,scope:decision.scope,polygon:decision.polygon})}
 const cfg={...asset.settings},gap=decision.gap;let ui={...asset.project.ui};
 if(asset.media_id!==gap.media_id)throw Error('Ce passage appartient à un autre média.');
 if(decision.kind==='temporal'){
  const fps=asset.project.media.fps||25,start=decision.scope==='trajectory'?0:decision.scope==='frame'?Math.round(decision.time*fps)/fps:gap.start,end=decision.scope==='trajectory'?Math.max(1/fps,asset.project.media.duration):decision.scope==='frame'?start+1/fps:gap.end;
  if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start||['link','split'].includes(decision.action)&&(!decision.target_id||subjectKey(decision.target_id)===subjectKey(gap.subject_id)))throw Error('Décision ou identité cible invalide.');
  const item:TemporalDecision={id:crypto.randomUUID(),subject_id:gap.subject_id,action:decision.action,start,end,...(decision.target_id?{target_id:decision.target_id}:{}),...(decision.polygon?{polygon:decision.polygon}:{})};
  const history=[...(cfg.temporal_decisions as TemporalDecision[]||[]),item];const graph=new Map<string,string[]>();for(const d of history)if(['link','split'].includes(d.action)&&d.target_id){const a=subjectKey(d.subject_id);graph.set(a,[...(graph.get(a)||[]),subjectKey(d.target_id)])}const visiting=new Set<string>(),done=new Set<string>();const visit=(n:string)=>{if(visiting.has(n))throw Error('Ce rattachement crée une boucle entre les identités.');if(done.has(n))return;visiting.add(n);for(const next of graph.get(n)||[])visit(next);visiting.delete(n);done.add(n)};for(const node of graph.keys())visit(node);
  cfg.temporal_decisions=history;
  if(item.action==='split'&&item.target_id){const source=reviewSubject(asset,gap);const overrides={...(cfg.subject_overrides as Record<string,unknown>||{})};const inherited=(cfg.subject_overrides as Record<string,{style?:unknown}>||{})[subjectKey(gap.subject_id)]?.style||(source.seed as Reference|undefined)?.style;overrides[item.target_id]={class:source.label+' · suite',color:source.color,...(inherited?{style:inherited}:{}),updated:Date.now()};cfg.subject_overrides=overrides;
   const id=item.target_id,time=item.start,box=decision.box||gap.reference_box||source.referenceBox||[10,10,20,30],candidate={id,class:source.label+' · suite',category:gap.category||'custom',color:source.color,time,box,sample:0};
   ui.candidates=[...(ui.candidates as unknown[]||[]),candidate];const numbers=[...(ui.numbers as [string,number][]||[])];numbers.push([id,Math.max(0,...numbers.map(z=>z[1]))+1]);ui.numbers=numbers;
   cfg.references=[...(cfg.references as unknown[]||[]),{...candidate,label:candidate.class,hidden:false}];cfg.selected_ids=[...new Set([...(cfg.selected_ids as string[]||[]),id])];
  }
 }
 else if(decision.kind==='absent')cfg.confirmed_gaps=[...new Set([...(cfg.confirmed_gaps as string[]||[]),gap.id])];
 else{
  const fps=asset.project.media.fps||25,t=Math.round(decision.time*fps)/fps,box=decision.box;
  if(!Number.isFinite(t)||t<gap.start-1/fps||t>=gap.end+1e-6||box.length!==4||box.some(v=>!Number.isFinite(v)||v<0||v>100)||box[2]<=0||box[3]<=0||box[0]+box[2]>100.001||box[1]+box[3]>100.001)throw Error('Tracez un cadre valide sur une image du passage.');
  const source=((cfg.temporal_decisions||[]) as TemporalDecision[]).find(d=>d.action==='split'&&subjectKey(d.target_id||'')===subjectKey(gap.subject_id))?.subject_id||gap.subject_id;const info=reviewSubject(asset,{...gap,subject_id:source}),custom=[...(cfg.custom as Reference[]||[])],index=custom.findIndex(s=>subjectKey(s.id)===subjectKey(source));
  const base=custom[index]||info.seed||{id:gap.subject_id,label:gap.label,time:gap.reference_time??t,box:gap.reference_box||box,color:info.color,category:gap.category};
  const same=Math.abs(base.time-t)<.5/fps;
  const seed={...base,id:source,hidden:base.hidden??false,...(same?{box}:{anchors:[...(base.anchors||[]).filter(a=>Math.abs(a.time-t)>=.5/fps),{time:t,box}]})};
  if(index<0)custom.push(seed);else custom[index]=seed;
  cfg.custom=custom;
 }
 return {...asset,settings:cfg,project:{...asset.project,settings:cfg,ui}};
}
