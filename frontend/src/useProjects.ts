import {useEffect,useMemo,useRef,useState} from 'react';
import type {ProjectMode,SavedProject} from './ProjectControls';
export type ProjectData={format:string;version:number;id:string|null;name:string;media_id:string;settings:Record<string,unknown>;ui:Record<string,unknown>;stamp?:number;media?:Record<string,unknown>};
declare global{interface Window{pywebview?:{api:{finish_close:()=>Promise<boolean>}}}}
async function request<T>(path:string,body?:unknown):Promise<T>{const controller=new AbortController(),timer=window.setTimeout(()=>controller.abort(),8000);try{const response=await fetch('/api/projects'+path,{signal:controller.signal,...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});const text=await response.text();let data;try{data=JSON.parse(text)}catch{throw new Error('Le moteur ne répond pas correctement. Relancez Privex Studio.')};if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Projet invalide ou moteur indisponible.');return data}catch(e){if(e instanceof TypeError)throw new Error('Connexion au moteur local interrompue. Votre copie locale du projet est conservée. Réessayez dans un instant.');if((e as Error).name==='AbortError')throw new Error('La reprise dépasse 8 secondes. Vous pouvez réessayer ou ouvrir un nouveau projet ; votre copie est conservée.');throw e}finally{window.clearTimeout(timer)}}
export function useProjects({state,editKey,fallbackName,busy:taskBusy,onRestore,onCancelTask,onClear,onNotice}:{state:{media_id:string;settings:Record<string,unknown>;ui:Record<string,unknown>}|null;editKey:string;fallbackName:string;busy:boolean;onRestore:(data:ProjectData)=>Promise<void>;onCancelTask:()=>Promise<void>;onClear:()=>void;onNotice:(s:string)=>void}){
 const [id,setId]=useState<string|null>(null),[name,setName]=useState(''),[savedKey,setSavedKey]=useState(''),[mode,setMode]=useState<ProjectMode|null>(null),[list,setList]=useState<SavedProject[]>([]),[recovery,setRecovery]=useState<ProjectData|null>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false),[restoreToken,setRestoreToken]=useState(0);
 const forceNew=useRef(false),allowClose=useRef(false),pauseDraft=useRef(false),draftTimer=useRef<number|undefined>(undefined),draftQueue=useRef<Promise<unknown>>(Promise.resolve()),cleanRestore=useRef(false),input=useRef<HTMLInputElement>(null);
 const pendingOpen=useRef<(()=>Promise<void>)|null>(null);const [replacing,setReplacing]=useState(false);
 const dirty=!!state&&savedKey!==editKey;
 const displayName=name||fallbackName||'Mon projet';
 const snapshot:ProjectData|null=state?{...state,format:'videomask-project',version:1,id,name:displayName,ui:{...state.ui,dirty}}:null;
 const latest=useRef({snapshot,editKey,dirty,onRestore,onCancelTask,onClear,onNotice});latest.current={snapshot,editKey,dirty,onRestore,onCancelTask,onClear,onNotice};
 useEffect(()=>{if(cleanRestore.current){cleanRestore.current=false;setSavedKey(editKey)}},[editKey,restoreToken]);
 const serialize=useMemo(()=>JSON.stringify(snapshot),[state,id,name,displayName,dirty]);
 useEffect(()=>{
  if(!snapshot||pauseDraft.current)return;
  // This synchronous copy also protects the browser fallback at beforeunload.
  try{localStorage.setItem('videomask-project-recovery',JSON.stringify({...JSON.parse(serialize),stamp:Date.now()}))}catch{}
  draftTimer.current=window.setTimeout(()=>{if(pauseDraft.current)return;const copy={...JSON.parse(serialize),stamp:Date.now()};draftQueue.current=draftQueue.current.catch(()=>{}).then(()=>request('/draft',copy)).catch(()=>{})},500);
  return()=>window.clearTimeout(draftTimer.current)
 },[serialize]);
 useEffect(()=>{let disposed=false;let local:ProjectData|null=null;try{const raw=localStorage.getItem('videomask-project-recovery');if(raw)local=JSON.parse(raw)}catch{}void request<ProjectData>('/draft').catch(()=>local).then(data=>{const newest=local&&(!data||(local.stamp||0)>(data.stamp||0))?local:data;if(newest&&!disposed){setRecovery(newest);setMode('recover')}});return()=>{disposed=true}},[]);
 const save=async(title=displayName,asNew=false)=>{
  const current=latest.current;if(!current.snapshot)return null;
  const result=await request<SavedProject>('',{...current.snapshot,id:asNew?null:id,name:title,stamp:Date.now()});setId(result.id);setName(result.name);setSavedKey(current.editKey);current.onNotice('Projet sauvegardé.');return result;
 };
 const discardRecovery=async()=>{await draftQueue.current.catch(()=>{});await request('/draft/discard',{});localStorage.removeItem('videomask-project-recovery')};
 const finishClose=async()=>{
  const current=latest.current;pauseDraft.current=true;window.clearTimeout(draftTimer.current);
  if(taskBusy)await current.onCancelTask();
  if(current.snapshot)await discardRecovery().catch(()=>{});
  allowClose.current=true;
  if(window.pywebview?.api)await window.pywebview.api.finish_close();
  else {current.onClear();setMode(null);setId(null);setName('');setSavedKey('');current.onNotice('Projet fermé. Vous pouvez fermer cette fenêtre.');pauseDraft.current=false;allowClose.current=false}
 };
 const requestClose=()=>{pendingOpen.current=null;setReplacing(false);forceNew.current=false;setError('');if(latest.current.dirty)setMode('close');else void finishClose()};
 const closeRef=useRef(requestClose);closeRef.current=requestClose;
 useEffect(()=>{
  const native=()=>closeRef.current();
  const browser=(e:BeforeUnloadEvent)=>{const current=latest.current;if(!current.dirty||allowClose.current||window.pywebview?.api)return;try{localStorage.setItem('videomask-project-recovery',JSON.stringify({...current.snapshot,stamp:Date.now()}))}catch{}e.preventDefault();e.returnValue='';};
  window.addEventListener('videomask:close-request',native);window.addEventListener('beforeunload',browser);
  return()=>{window.removeEventListener('videomask:close-request',native);window.removeEventListener('beforeunload',browser)}
 },[]);
 const restore=async(data:ProjectData,clean:boolean)=>{
  pauseDraft.current=true;window.clearTimeout(draftTimer.current);cleanRestore.current=clean;
  await latest.current.onRestore(data);setId(data.id||null);setName(data.name);setMode(null);setRecovery(null);setRestoreToken(v=>v+1);pauseDraft.current=false;allowClose.current=false;
  latest.current.onNotice('Projet rouvert avec ses sujets et ses réglages.');
 };
 const performLoad=async(projectId:string)=>{setBusy(true);setError('');try{await restore(await request<ProjectData>('/'+projectId),true)}catch(e){setError((e as Error).message);cleanRestore.current=false;pauseDraft.current=false}finally{setBusy(false)}};
 const load=async(projectId:string)=>{if(latest.current.dirty){pendingOpen.current=()=>performLoad(projectId);forceNew.current=false;setReplacing(true);setMode('close');return;}await performLoad(projectId)};
 const recover=async()=>{if(!recovery)return;setBusy(true);setError('');try{await restore(await request<ProjectData>('/import',recovery),!recovery.ui?.dirty)}catch(e){setError((e as Error).message);pauseDraft.current=false;cleanRestore.current=false}finally{setBusy(false)}};
 const continueOpen=async()=>{const action=pendingOpen.current;pendingOpen.current=null;setReplacing(false);pauseDraft.current=true;window.clearTimeout(draftTimer.current);await discardRecovery();if(action)await action();pauseDraft.current=false;};
 const discard=async()=>{setBusy(true);setError('');try{if(mode==='recover'){pauseDraft.current=true;await discardRecovery();setRecovery(null);setMode(null);pauseDraft.current=false}else if(pendingOpen.current)await continueOpen();else await finishClose()}catch(e){setError((e as Error).message);pauseDraft.current=false}finally{setBusy(false)}};
 const saveDialog=async(title:string)=>{setBusy(true);setError('');try{await save(title,forceNew.current);if(mode==='close'){if(pendingOpen.current)await continueOpen();else await finishClose()}else setMode(null)}catch(e){setError((e as Error).message);pauseDraft.current=false}finally{setBusy(false)}};
 const open=async()=>{setMode('open');setError('');try{setList(await request<SavedProject[]>(''))}catch(e){setError((e as Error).message)}};
 const saveMenu=()=>{forceNew.current=false;setError('');if(id){setBusy(true);void save().catch(e=>latest.current.onNotice(e.message)).finally(()=>setBusy(false))}else setMode('save')};
 const saveAs=()=>{forceNew.current=true;setError('');setMode('save')};
 const download=async()=>{setBusy(true);try{const saved=await save();if(saved){const a=document.createElement('a');a.href='/api/projects/'+saved.id+'/download';a.download=displayName+'.videomask';a.click()}}catch(e){latest.current.onNotice((e as Error).message)}finally{setBusy(false)}};
 const importFile=async(file:File)=>{setBusy(true);setError('');try{if(file.size>5*1024**2)throw new Error('Fichier de projet trop volumineux.');const data=await request<ProjectData>('/import',JSON.parse(await file.text()));if(latest.current.dirty){pendingOpen.current=()=>restore(data,true);forceNew.current=false;setReplacing(true);setMode('close')}else await restore(data,true)}catch(e){setError((e as Error).message);pauseDraft.current=false;cleanRestore.current=false}finally{setBusy(false)}};
 const reset=()=>{setId(null);setName('');setSavedKey('');pauseDraft.current=false;allowClose.current=false};
 return {id,name:displayName,dirty,mode,list,busy,error,input,reset,open,saveMenu,saveAs,download,requestClose,saveDialog,discard,load,recover,replacing,cancel:()=>{pendingOpen.current=null;setReplacing(false);setMode(null);setError('');pauseDraft.current=false},recoveryName:recovery?.name,importFile};
}
