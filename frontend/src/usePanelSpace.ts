import {useEffect,useRef,useState,type RefObject} from 'react';
/** Borrow only the timeline's height; authoring state and the timeline layout stay intact. */
export function usePanelSpace(workspace:RefObject<HTMLDivElement|null>,enabled:boolean,layoutKey:boolean,panelId:'inspector-panel'|'library-panel'){
 const selector='#'+panelId;
 const [extension,setExtension]=useState(0),[engaged,setEngaged]=useState(false);const engagedRef=useRef(false),held=useRef(false);
 const engage=(next:boolean)=>{if(engagedRef.current===next)return;engagedRef.current=next;setEngaged(next)};
 useEffect(()=>{const area=workspace.current,timeline=document.getElementById('timeline-panel');if(!area||!timeline||!enabled){setExtension(0);engage(false);return}
  const measure=()=>{const bounds=area.getBoundingClientRect(),end=timeline.getBoundingClientRect().bottom;const extra=window.innerWidth>620&&timeline.getAttribute('aria-hidden')!=='true'?Math.max(0,end-bounds.bottom+8):0;setExtension(extra)};
  measure();const observer=new ResizeObserver(measure);observer.observe(area);observer.observe(timeline);window.addEventListener('resize',measure);return()=>{observer.disconnect();window.removeEventListener('resize',measure)};
 },[workspace,enabled,layoutKey]);
 useEffect(()=>{if(!enabled)return;const inside=(target:EventTarget|null)=>target instanceof Element&&!!target.closest(selector);
  const pointer=(e:PointerEvent)=>{if(e.pointerType==='touch'||e.buttons||held.current)return;const area=workspace.current;if(!area)return;const targetInside=inside(e.target);if(!targetInside){if(!document.querySelector(selector+' dialog[open]'))engage(false);return}if(engagedRef.current||e.clientY<=area.getBoundingClientRect().bottom)engage(true)};
  const down=(e:PointerEvent)=>{held.current=inside(e.target);engage(held.current)};
  const up=(e:PointerEvent)=>{held.current=false;engage(inside(e.target))};
  const focus=(e:FocusEvent)=>engage(inside(e.target));
  const leave=()=>{if(!held.current&&!document.querySelector(selector+' dialog[open]'))engage(false)};
  document.addEventListener('pointermove',pointer,true);document.addEventListener('pointerover',pointer,true);document.addEventListener('pointerdown',down,true);document.addEventListener('pointerup',up,true);document.addEventListener('pointercancel',up,true);document.addEventListener('focusin',focus,true);document.documentElement.addEventListener('pointerleave',leave);
  return()=>{document.removeEventListener('pointermove',pointer,true);document.removeEventListener('pointerover',pointer,true);document.removeEventListener('pointerdown',down,true);document.removeEventListener('pointerup',up,true);document.removeEventListener('pointercancel',up,true);document.removeEventListener('focusin',focus,true);document.documentElement.removeEventListener('pointerleave',leave);held.current=false;engage(false)};
 },[workspace,enabled,selector]);
 return {engaged:enabled&&engaged,expanded:enabled&&engaged&&extension>1,extension};
}
