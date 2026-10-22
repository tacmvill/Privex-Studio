import {useLayoutEffect,useRef,type RefObject} from 'react';
import type {PreviewMask} from './useTrackedBoxes';

/** Native video pixels + cached geometry. Missing data stays covered, never flashes source. */
export function paintMasks(canvas:HTMLCanvasElement,source:CanvasImageSource,width:number,height:number,masks:PreviewMask[],defaults:Record<string,unknown>,renderScale=1){
 if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;
 const ctx=canvas.getContext('2d');if(!ctx)return;
 ctx.drawImage(source,0,0,width,height);
 for(const mask of masks){
  const s={...defaults,...mask.style};if(s.effect==='pixels_noirs')s.shape='pixelise';const fullBody=mask.category==='personne'||mask.class==='personne';const factor=mask.privacy_coverage||fullBody?Math.max(1,Number(s.size??100)/100):Number(s.size??100)/100;
  const raw=mask.box.map((v,i)=>v/100*(i%2?height:width));let [x,y,w,h]=raw;const margin=Number(s.margin??0)/100;
  x-=w*(factor-1)/2;y-=h*(factor-1)/2;w*=factor;h*=factor;x-=w*margin;y-=h*margin;w*=1+2*margin;h*=1+2*margin;
  const right=Math.min(width,Math.ceil(x+w)),bottom=Math.min(height,Math.ceil(y+h));x=Math.max(0,Math.floor(x));y=Math.max(0,Math.floor(y));w=right-x;h=bottom-y;if(w<1||h<1)continue;
  const alpha=document.createElement('canvas');alpha.width=w;alpha.height=h;const shape=alpha.getContext('2d')!;shape.fillStyle='white';shape.beginPath();
  const contour=s.shape!=='rectangle'&&mask.polygons?.length;
  if(contour){const cx=raw[0]+raw[2]/2,cy=raw[1]+raw[3]/2;for(const p of mask.polygons!){p.forEach(([px,py],i)=>{const tx=cx+(px-cx)*factor-x,ty=cy+(py-cy)*factor-y;i?shape.lineTo(tx,ty):shape.moveTo(tx,ty)});shape.closePath()}shape.fill();
   const dilate=Math.round(Math.max(raw[2],raw[3])*margin/2);if(dilate){shape.lineWidth=dilate*2;shape.strokeStyle='white';shape.lineJoin='round';shape.stroke()}
  }else shape.fillRect(0,0,w,h);
  if(contour&&fullBody){shape.beginPath();shape.ellipse(raw[0]+raw[2]*.5-x,raw[1]+raw[3]*.5-raw[3]*.34*factor-y,Math.max(1,raw[2]*.52*factor),Math.max(1,raw[3]*.19*factor),0,0,Math.PI*2);shape.fill()}
  if(contour&&s.shape==='pixelise'){const nativeBlock=['pixels_noirs','pixels_gris','pixels_rgb'].includes(String(s.effect))?Math.max(2,Math.min(128,Math.round((2+(s.tile_size!=null?Number(s.tile_size)*1.26:Number(s.intensity??70)*.32))/2)*2)):Math.max(4,Math.round(4+Number(s.intensity??70)*.15)),block=Math.max(1,Math.round(nativeBlock*renderScale)),pixels=shape.getImageData(0,0,w,h).data;shape.clearRect(0,0,w,h);for(let by=0;by<h;by+=block)for(let bx=0;bx<w;bx+=block){let found=false;for(let yy=by;yy<Math.min(h,by+block)&&!found;yy++)for(let xx=bx;xx<Math.min(w,bx+block);xx++)if(pixels[(yy*w+xx)*4+3]){found=true;break}if(found)shape.fillRect(bx,by,block,block)}}
  const feather=mask.privacy_coverage?0:Number(s.feather??0);let alphaSource:HTMLCanvasElement=alpha;
  if(feather){const softened=document.createElement('canvas');softened.width=w;softened.height=h;const sc=softened.getContext('2d')!;sc.filter=`blur(${Math.max(1,Math.round(feather*.08*renderScale))}px)`;sc.drawImage(alpha,0,0);alphaSource=softened}
  const effect=document.createElement('canvas');effect.width=w;effect.height=h;const ec=effect.getContext('2d')!;
  if(s.effect==='noir'||s.effect==='pixels_noirs'){ec.fillStyle='black';ec.fillRect(0,0,w,h)}else{
   const intensity=Number(s.intensity??70),pixel=['pixels','pixels_gris','pixels_rgb'].includes(String(s.effect)),block=Math.max(1,Math.round(Math.max(2,Math.round((2+(s.tile_size!=null?Number(s.tile_size)*1.26:intensity*.32))/2)*2)*renderScale)),tile=document.createElement('canvas');tile.width=pixel?Math.max(1,Math.floor(w/block)):w;tile.height=pixel?Math.max(1,Math.floor(h/block)):h;const tc=tile.getContext('2d')!;tc.drawImage(canvas,x,y,w,h,0,0,tile.width,tile.height);
   if(s.effect==='pixels_gris'||s.effect==='pixels_rgb'){const image=tc.getImageData(0,0,tile.width,tile.height),data=image.data,hex=/^#[0-9a-f]{6}$/i.test(String(s.mosaic_color))?String(s.mosaic_color):'#007aff',rgb=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));for(let i=0;i<data.length;i+=4){const light=Math.round(data[i]*.299+data[i+1]*.587+data[i+2]*.114);if(s.effect==='pixels_gris'){const level=light<85?0:light<170?128:255;data[i]=data[i+1]=data[i+2]=level}else for(let c=0;c<3;c++)data[i+c]=Math.round(light*rgb[c]/255)}tc.putImageData(image,0,0)}
   ec.fillStyle='black';ec.fillRect(0,0,w,h);ec.imageSmoothingEnabled=!pixel;ec.filter=pixel?'none':`blur(${Math.max(.5,intensity*(s.effect==='gaussien'?.6:.3)*renderScale)}px)`;ec.drawImage(tile,0,0,tile.width,tile.height,0,0,w,h);ec.filter='none';
  }
  ec.globalCompositeOperation='destination-in';ec.drawImage(alphaSource,0,0);ctx.drawImage(effect,x,y);
 }

}

export function LiveMaskCanvas({video,width,height,masks,time,settings,filter,playing=false,lookupMasks}:{lookupMasks?:(time:number)=>PreviewMask[]|undefined;video:RefObject<HTMLVideoElement|null>;width:number;height:number;masks?:PreviewMask[];time:number;settings:Record<string,unknown>;filter?:string;playing?:boolean}){
 const canvas=useRef<HTMLCanvasElement>(null),painted=useRef(false);
 useLayoutEffect(()=>{const el=canvas.current,v=video.current;if(!el)return;
  const paint=(frameTime=time)=>{
   const frameMasks=lookupMasks?lookupMasks(frameTime):masks;
   // Retain the last fully covered image during a brief buffer refill. The transport waits too.
   if(!v||v.readyState<2||frameMasks===undefined){if(!painted.current){const ctx=el.getContext('2d');if(ctx){ctx.fillStyle='#18181b';ctx.fillRect(0,0,el.width,el.height)}}return}
   const view=el.getBoundingClientRect(),budget=Math.max(960,Math.min(1920,Math.max(view.width,view.height)*(window.devicePixelRatio||1))),scale=playing?Math.min(1,budget/Math.max(width,height)):1;
   const geometry=scale===1?frameMasks:frameMasks.map(m=>({...m,polygons:m.polygons?.map(p=>p.map(([x,y])=>[x*scale,y*scale]))}));
   paintMasks(el,v,Math.max(1,Math.round(width*scale)),Math.max(1,Math.round(height*scale)),geometry,settings,scale);painted.current=true;
  };const initial=()=>paint(playing&&v?v.currentTime:time);initial();v?.addEventListener('loadeddata',initial);let active=true,handle=0;const tick=(_now:number,meta?:VideoFrameCallbackMetadata)=>{if(!active||!v)return;paint(meta?.mediaTime??v.currentTime);handle=v.requestVideoFrameCallback?v.requestVideoFrameCallback(tick):requestAnimationFrame(tick)};if(playing&&v)handle=v.requestVideoFrameCallback?v.requestVideoFrameCallback(tick):requestAnimationFrame(tick);return()=>{active=false;v?.removeEventListener('loadeddata',initial);if(v?.cancelVideoFrameCallback)v.cancelVideoFrameCallback(handle);else cancelAnimationFrame(handle)};
 },[time,masks,settings,width,height,playing]);
 return <><canvas ref={canvas} width={width} height={height} aria-label="Aperçu du masquage en lecture" style={{position:'absolute',inset:0,width:'100%',height:'100%',pointerEvents:'none',filter}}/>{masks===undefined&&<span role="status" style={{position:'absolute',inset:0,display:'grid',placeItems:'center',color:'white',fontSize:12,padding:16,pointerEvents:'none'}}>Préparation de la lecture masquée…</span>}</>;
}
