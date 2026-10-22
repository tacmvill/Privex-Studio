import {useLayoutEffect,useRef,type RefObject} from 'react';
import {subjectColor} from './subjectColor';
import type {TrackedBox} from './useTrackedBoxes';

/** Position DOM boxes in the callback for the frame actually presented by video. */
export function FrameTrackingOverlay({video,playing,time,offset=0,lookup,selected,covered,onPick,disabled,showBoxes=true,hiddenIds=[]}:{video:RefObject<HTMLVideoElement|null>;playing:boolean;time:number;offset?:number;lookup:(time:number)=>{boxes?:TrackedBox[];masks?:{id:string}[]};selected:Set<string>;covered:boolean;onPick?:(box:TrackedBox,time:number)=>void;disabled:boolean;showBoxes?:boolean;hiddenIds?:string[]}){
 const host=useRef<HTMLDivElement>(null),latest=useRef({lookup,selected,onPick,disabled,covered,hiddenIds});latest.current={lookup,selected,onPick,disabled,covered,hiddenIds};
 const nodes=useRef(new Map<string,{node:HTMLButtonElement;label:HTMLSpanElement;box:TrackedBox}>());
 const paint=(at:number)=>{const root=host.current;if(!root)return;root.dataset.frameTime=String(at);const value=latest.current.lookup(at),present=new Set<string>(),clean=(s:string)=>s.replace(/^m-da/,'a'),masked=new Set((value.masks||[]).map(m=>clean(m.id))),hidden=new Set(latest.current.hiddenIds.map(clean));
  for(const [rank,box] of [...(value.boxes||[])].sort((a,b)=>b.box[2]*b.box[3]-a.box[2]*a.box[3]).entries()){if([box.id,...(box.reference_ids||[])].some(id=>hidden.has(clean(id))))continue;present.add(box.id);let entry=nodes.current.get(box.id);
   if(!entry){const node=document.createElement('button'),label=document.createElement('span');node.type='button';node.className='detection-box';node.dataset.subjectId=box.id;node.append(label);entry={node,label,box};nodes.current.set(box.id,entry);node.onclick=e=>{e.stopPropagation();const current=nodes.current.get(box.id);if(current&&!latest.current.disabled)latest.current.onPick?.(current.box,Number(root.dataset.frameTime)||0)};node.onkeydown=e=>{if(e.key==='Enter'||e.key===' ')e.stopPropagation()};root.append(node)}
   entry.box=box;const ids=[box.id,...(box.reference_ids||[])].map(clean),chosen=ids.some(id=>latest.current.selected.has(id)),active=latest.current.covered&&chosen||ids.some(id=>masked.has(id)),color=chosen?(box.id.startsWith('a')||box.id.startsWith('m-da')?subjectColor(box.id):box.color):'#c7c7cc';
   Object.assign(entry.node.style,{position:'absolute',left:box.box[0]+'%',top:box.box[1]+'%',width:box.box[2]+'%',height:box.box[3]+'%',borderColor:color,display:''});entry.node.style.setProperty('--subject-color',color);entry.node.dataset.maskActive=String(active);entry.node.disabled=disabled;entry.node.style.zIndex=String(rank+1);entry.node.setAttribute('aria-label',`Afficher les informations de ${box.class}`);entry.node.title=`${box.class} · Cliquer pour les informations et les actions`;entry.label.textContent=box.class+(chosen&&!active?' · Non masqué':'');
  }
  for(const [id,entry] of nodes.current)if(!present.has(id)){entry.node.remove();nodes.current.delete(id)}
 };
 const paintRef=useRef(paint);paintRef.current=paint;
 useLayoutEffect(()=>{if(!playing)paintRef.current(time)},[time,lookup,playing,selected,covered,disabled,showBoxes,hiddenIds]);
 useLayoutEffect(()=>{const v=video.current;if(!playing||!v)return;let alive=true,id=0;const tick=(_now:number,meta?:VideoFrameCallbackMetadata)=>{if(!alive)return;paintRef.current((meta?.mediaTime??v.currentTime)+offset);id=v.requestVideoFrameCallback?v.requestVideoFrameCallback(tick):requestAnimationFrame(tick)};id=v.requestVideoFrameCallback?v.requestVideoFrameCallback(tick):requestAnimationFrame(tick);return()=>{alive=false;if(v.cancelVideoFrameCallback)v.cancelVideoFrameCallback(id);else cancelAnimationFrame(id)}},[playing,video,offset]);
 return <div ref={host} className="sequence-tracked-boxes" data-show-boxes={showBoxes} aria-label="Sujets suivis dans le montage"/>;
}
