import {useEffect,useRef} from 'react';
import {CloseGlyph} from './SFIcon';
export function FinalPreview({open,url,muted,onClose}:{open:boolean;url:string;muted:boolean;onClose:()=>void;}){
 const dialog=useRef<HTMLDialogElement>(null),video=useRef<HTMLVideoElement>(null);
 useEffect(()=>{if(open&&!dialog.current?.open)dialog.current?.showModal();if(!open){video.current?.pause();dialog.current?.close()}},[open]);
 return <dialog ref={dialog} className="final-preview-dialog" aria-label="Résultat complet du montage" onCancel={e=>{e.preventDefault();onClose()}}><header><div><strong>Résultat complet du montage</strong><p>Regardez les masques, les ajouts et le son puis utilisez le bouton Exporter en haut de l’application.</p></div><button className="close-button" aria-label="Fermer la prévisualisation complète" onClick={onClose}><CloseGlyph/></button></header><video ref={video} src={url} controls preload="metadata" muted={muted} aria-label="Vidéo finale masquée"/><footer><span>Les passages incertains ont été contrôlés avant ce rendu.</span><button onClick={onClose}>Revenir aux réglages</button><button className="btn-primary" onClick={onClose}>Terminer la prévisualisation</button></footer></dialog>;
}
