import type {SequenceAsset} from './sequence';
import {subjectKey,type TrackingGap} from './trackingReviewModel';
export type SubjectState='visible'|'candidate'|'doubt'|'occluded'|'absent'|'out_of_frame'|'rejected';
export type Trajectory={id:string;label:string;category:string;color:string;existence:string;reference_time?:number;reference_box?:number[];intervals:{state:SubjectState;start:number;end:number;first:number;last:number}[]};
export type TemporalDecision={id:string;subject_id:string;action:'confirm'|'reject'|'suppress'|'absent'|'occluded'|'link'|'split'|'contour';polygon?:number[][];start:number;end:number;target_id?:string};
export const stateLabel:Record<SubjectState,string>={visible:'Visible et suivi',candidate:'Détection à confirmer',doubt:'Décision nécessaire',occluded:'Caché · masque suspendu',absent:'Absent · aucun masque',out_of_frame:'Hors champ',rejected:'Faux positif rejeté'};
export function decisionsAt(asset:Pick<SequenceAsset,'settings'>,id:string,time:number){return ((asset.settings.temporal_decisions||[]) as TemporalDecision[]).filter(d=>subjectKey(d.subject_id)===subjectKey(id)&&d.start<=time+1e-7&&time<d.end-1e-7)}
export function decisionResolves(asset:{settings:Record<string,unknown>;project:{media:{fps:number}}},gap:TrackingGap){
 const fps=asset.project.media.fps||25;
 for(let i=Math.round(gap.start*fps);i<Math.round(gap.end*fps);i++){
  if(!decisionsAt(asset,gap.subject_id,i/fps).some(d=>['reject','suppress','absent','occluded'].includes(d.action)||['confirm','contour'].includes(d.action)&&gap.kind==='detection'||['link','split'].includes(d.action)&&gap.kind==='identity'))return false;
 }return true;
}
export function renderedKey(data:{assets:SequenceAsset[];clips:unknown[];texts?:unknown[];images?:unknown[]},step:number){return JSON.stringify({assets:data.assets.map(a=>({id:a.media_id,settings:a.settings})),clips:data.clips,texts:data.texts,images:data.images,step})}

export function displayTrajectories(asset:SequenceAsset):Trajectory[]{
 const source=(asset.project.ui.trajectories||[]) as Trajectory[],result=new Map<string,Trajectory>(),all=(asset.settings.temporal_decisions||[]) as TemporalDecision[];
 for(const track of source)for(const interval of track.intervals){
  const relevant=all.filter(d=>subjectKey(d.subject_id)===subjectKey(track.id)&&d.start<interval.end&&d.end>interval.start);
  const bounds=[...new Set([interval.start,interval.end,...relevant.flatMap(d=>[Math.max(interval.start,d.start),Math.min(interval.end,d.end)])])].sort((a,b)=>a-b);
  for(let i=0;i<bounds.length-1;i++){
   const start=bounds[i],end=bounds[i+1],mid=(start+end)/2;let id=track.id,state=interval.state;
   for(const decision of decisionsAt(asset,id,mid)){
    if(['link','split'].includes(decision.action)&&decision.target_id)id=decision.target_id;
    else if(decision.action==='reject')state='rejected';else if(['suppress','absent'].includes(decision.action))state='absent';else if(decision.action==='occluded')state='occluded';else if(decision.action==='confirm'&&state==='candidate')state='visible';
   }
   const override=(asset.settings.subject_overrides as Record<string,{class?:string;color?:string}>||{})[id];const entry=result.get(id)||{...track,id,label:override?.class||track.label,color:override?.color||track.color,intervals:[]};entry.intervals.push({...interval,start,end,state});result.set(id,entry);
  }
 }return [...result.values()];
}
