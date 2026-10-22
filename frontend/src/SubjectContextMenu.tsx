import {useEffect,useRef} from 'react';
import {SFIcon} from './SFIcon';
export type SubjectMenuTarget={id:string;label:string;x:number;y:number};
export function SubjectContextMenu({target,onDelete,onClose}:{target:SubjectMenuTarget;onDelete:()=>void;onClose:()=>void}){
 const ref=useRef<HTMLDivElement>(null);
 useEffect(()=>{const node=ref.current;if(!node)return;node.showPopover();const rect=node.getBoundingClientRect();node.style.left=`${Math.max(8,Math.min(target.x,innerWidth-rect.width-8))}px`;node.style.top=`${Math.max(8,Math.min(target.y,innerHeight-rect.height-8))}px`;node.querySelector<HTMLButtonElement>('button')?.focus({preventScroll:true});return()=>{if(node.matches(':popover-open'))node.hidePopover()}},[target]);
 return <div ref={ref} popover="auto" role="menu" aria-label={`Actions de ${target.label}`} className="subject-context-menu" onToggle={e=>{if(e.newState==='closed')onClose()}}><button type="button" role="menuitem" onClick={onDelete}><SFIcon name="delete"/>Supprimer ce sujet</button></div>;
}
