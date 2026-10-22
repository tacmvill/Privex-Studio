import {PowerStatus,type PowerSupply} from './PowerStatus';
import {useEffect,useRef,useState} from 'react';
import {SFIcon,CloseGlyph} from './SFIcon';
import {PerformanceStatus,type PerformanceData} from './PerformanceStatus';
export type TaskState={id:string;phase:string;progress:number;elapsed:number;performance?:PerformanceData};
type Temperature={available:boolean;celsius:number|null;sensor_id:string|null;label:string;source:string;sensors:{id:string;label:string;celsius:number}[]};
type Resources={temperature?:Temperature;power?:PowerSupply;ready:boolean;sampled_at:number;cpu_percent:number;gpu_engine_percent:number|null;ram_percent:number;ram_used_mb:number;ram_total_mb:number};
const duration=(s:number)=>s<60?`${Math.ceil(s)} s`:`${Math.floor(s/60)} min ${Math.ceil(s%60)} s`;
const percent=(n:number|null|undefined)=>n==null?'Indisponible':`${Math.round(n)} %`;
function Chart({values,label,color,max=100}:{values:(number|null)[];label:string;color:string;max?:number}){
 const paths:string[]=[];let path='';
 values.forEach((v,i)=>{if(v==null){if(path)paths.push(path);path=''}else path+=`${path?'L':'M'}${(i*180/59).toFixed(1)},${(39-Math.min(max,Math.max(0,v))/max*36).toFixed(1)} `});if(path)paths.push(path);
 return <svg className="resource-chart" viewBox="0 0 180 42" role="img" aria-label={label}><path d="M0 39H180 M0 21H180 M0 3H180" className="chart-grid"/>{paths.map((d,i)=><path key={i} d={d} fill="none" stroke={color} strokeWidth="1.7" strokeLinejoin="round"/>)}</svg>;
}
export function TaskProgress({task,onCancel,lastPerformance,profile}:{task:TaskState|null;onCancel:()=>void;lastPerformance?:PerformanceData;profile:string}){
 const [open,setOpen]=useState(false),[resources,setResources]=useState<Resources|null>(null),[history,setHistory]=useState<Resources[]>([]);
 const ref=useRef<HTMLDivElement>(null),scrollPosition=useRef(0);const [diagnosticOpen,setDiagnosticOpen]=useState(false);
 useEffect(()=>{
  if(!open)return;
  const close=(e:PointerEvent)=>{if(!ref.current?.contains(e.target as Node))setOpen(false)};
  const key=(e:KeyboardEvent)=>{if(e.key==='Escape'){setOpen(false);ref.current?.querySelector('button')?.focus()}};
  document.addEventListener('pointerdown',close);document.addEventListener('keydown',key);
  return()=>{document.removeEventListener('pointerdown',close);document.removeEventListener('keydown',key)};
 },[open]);
 useEffect(()=>{
  let cancelled=false,timer:number;
  const poll=async()=>{try{const r=await fetch('/api/system/performance');if(r.ok){const value:Resources=await r.json();if(!cancelled&&value.ready){setResources(value);setHistory(h=>h.at(-1)?.sampled_at===value.sampled_at?h:[...h,value].slice(-60))}}}catch{/* Never invent a hardware measurement. */}finally{if(!cancelled)timer=window.setTimeout(poll,1000)}};
  void poll();return()=>{cancelled=true;window.clearTimeout(timer);};
 },[]);
 const data=task?task.performance:lastPerformance;
 const progress=Math.max(0,Math.min(100,task?.progress||0));
 const remaining=task&&progress>=5?Math.round(task.elapsed*(100-progress)/progress):null;
 const estimate=remaining==null?'Estimation en cours…':`≈ ${duration(remaining)} restantes`;
 const fresh=resources&&Date.now()/1000-resources.sampled_at<5?resources:null;
 const health=data?.outcome==='error'?{tone:'error',text:'Erreur de traitement'}:fresh&&fresh.ram_percent>=95?{tone:'error',text:'Mémoire presque saturée'}:fresh&&(fresh.cpu_percent>=90||fresh.ram_percent>=85)?{tone:'warning',text:'Ressources très sollicitées'}:fresh?{tone:'ok',text:'Ressources disponibles'}:{tone:'unknown',text:'Mesures en attente'};
 const providers=[...new Set(Object.values(data?.models||{}).flatMap(m=>Object.values(m.executed_shapes||{}).length?Object.values(m.executed_shapes||{}):m.primary?[m.primary==='DmlExecutionProvider'?'DirectML (session)':m.primary]:[]))].map(p=>p==='DmlExecutionProvider'?'DirectML':p==='CPUExecutionProvider'?'CPU':p).join(' + ');
 return <div className="toolbar-performance">
  {task&&<div className="toolbar-task-status" title={`${task.phase} · ${Math.round(progress)} % · ${duration(task.elapsed)} écoulées · ${estimate}`}>
   <span className="compact-task-info" role="status"><i className="loading-spinner"/><b>{Math.round(progress)} %</b><span className="compact-task-eta">{remaining==null?'…':`≈ ${duration(remaining)}`}</span></span>
   <button type="button" disabled={!task.id} title="Annuler le traitement" aria-label="Annuler le traitement" onClick={onCancel}><CloseGlyph/></button>
   <span className="compact-progress-track"><progress aria-label={task.phase} max={100} value={progress}/></span>
  </div>}
  <div ref={ref} className="project-menu performance-menu">
   <button type="button" aria-label="Menu Performances" aria-expanded={open} aria-controls="performance-popover" onClick={()=>setOpen(v=>!v)}><svg className="performance-glyph" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 12V8m4 4V4m4 8V6m4 6V2" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg><span className="performance-label">Performances</span><span className="project-disclosure"><SFIcon name="chevron" size={open?8.5:10}/></span></button>
   {open&&<div id="performance-popover" ref={node=>{if(node)node.scrollTop=scrollPosition.current}} onScroll={e=>{scrollPosition.current=e.currentTarget.scrollTop}} className="performance-popover" role="region" aria-label="Performances du PC et du traitement">
    <header><strong>Performances</strong></header><p className={`health-indicator ${health.tone}`}><span aria-hidden="true">{health.tone==='error'?'!':health.tone==='warning'?'!':health.tone==='ok'?'✓':'—'}</span>{health.text}</p>
    <div className="resource-cards">{([
     ['CPU',fresh?.cpu_percent,history.map(r=>r.cpu_percent),'#64a8ff',100],
     ['GPU',fresh?.gpu_engine_percent,history.map(r=>r.gpu_engine_percent),'#b99aff',100],
     ['Mémoire',fresh?.ram_percent,history.map(r=>r.ram_percent),'#6ac9ad',100],
     ['Température',fresh?.temperature?.available?fresh.temperature.celsius:null,history.map(r=>r.temperature?.available?r.temperature.celsius:null),'#e6a558',150]
    ] as const).map(([label,value,values,color,max])=><section className="resource-card" key={label}><span>{label}</span><b>{label==='Température'?(value==null?'Indisponible':`${value.toFixed(1)} °C`):percent(value)}</b><Chart values={[...values]} max={max} label={`Historique ${label} · ${label==='Température'?'0 à 150 °C':'0 à 100 %'} · une mesure par seconde`} color={color}/></section>)}</div>
    <p className="resource-caption">{fresh?`${(fresh.ram_used_mb/1024).toFixed(1)} / ${(fresh.ram_total_mb/1024).toFixed(1)} Go de RAM · GPU : moteur le plus actif`:'Mesures matérielles indisponibles ou en cours de chargement.'}</p>
    <p className="thermal-caption">{fresh?.temperature?.available?<span title={`Capteur système fourni par le firmware, pas une mesure CPU/GPU dédiée. ${fresh.temperature.sensors.map(s=>`${s.id} : ${s.celsius.toFixed(1)} °C`).join(' · ')}`}>{fresh.temperature.label}{fresh.temperature.sensors.length>1?' · zone la plus chaude':''}</span>:fresh?.temperature?'Température : aucun capteur accessible à Windows.':'Température : lecture en cours…'}</p>
    <PowerStatus value={fresh?.power}/>
    <dl className="task-facts"><div><dt>Étape</dt><dd>{task?.phase||'Au repos'}</dd></div><div><dt>Temps restant</dt><dd>{task?estimate:'—'}</dd></div><div><dt>Temps écoulé</dt><dd>{task?duration(task.elapsed):'—'}</dd></div><div><dt>Puissance du moteur</dt><dd>{profile==='rapide'?'Rapide':profile==='extreme'?'Fidèle approfondi':'Fidèle'}</dd></div><div><dt>Exécution IA{!task&&data?' · dernière tâche':''}</dt><dd>{providers||'Aucun modèle actif'}</dd></div>{task&&data&&<><div><dt>Images traitées</dt><dd>{data.video_frames_processed} / {data.video_frames_total}</dd></div><div><dt>Débit global</dt><dd>{data.overall_fps.toFixed(1)} images/s</dd></div><div><dt>RAM du moteur</dt><dd>{data.resources.ram_mb==null?'Indisponible':`${Math.round(data.resources.ram_mb)} Mo`}</dd></div></>}</dl>
    {data&&<details open={diagnosticOpen} onToggle={e=>setDiagnosticOpen(e.currentTarget.open)} className="performance-advanced"><summary><span>Diagnostic de {task?'la tâche':'la dernière tâche'}</span><span className="project-disclosure"><SFIcon name="chevron" size={diagnosticOpen?8.5:10}/></span></summary><PerformanceStatus data={data}/></details>}
   </div>}
  </div>
 </div>;
}
