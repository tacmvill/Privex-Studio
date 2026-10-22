import {sfBackwardFrameFill} from '@bradleyhodges/sfsymbols/sfBackwardFrameFill';
import {sfForwardFrameFill} from '@bradleyhodges/sfsymbols/sfForwardFrameFill';
import {sfSquareAndArrowUp} from '@bradleyhodges/sfsymbols/sfSquareAndArrowUp';
import {sfPauseFill} from '@bradleyhodges/sfsymbols/sfPauseFill';
import {sfPlayFill} from '@bradleyhodges/sfsymbols/sfPlayFill';
import {sfLink} from '@bradleyhodges/sfsymbols/sfLink';
import {sfWaveform} from '@bradleyhodges/sfsymbols/sfWaveform';
import {sfScissors} from '@bradleyhodges/sfsymbols/sfScissors';
import {sfViewfinder} from '@bradleyhodges/sfsymbols/sfViewfinder';
import {sfPaintpalette} from '@bradleyhodges/sfsymbols/sfPaintpalette';
import {sfCircleBadgeQuestionmark} from '@bradleyhodges/sfsymbols/sfCircleBadgeQuestionmark';
import {sfChevronDown} from '@bradleyhodges/sfsymbols/sfChevronDown';
import {sfArrowUturnBackward} from '@bradleyhodges/sfsymbols/sfArrowUturnBackward';
import {sfArrowUturnForward} from '@bradleyhodges/sfsymbols/sfArrowUturnForward';
import {sfCheckmark} from '@bradleyhodges/sfsymbols/sfCheckmark';
import {sfSunMax} from '@bradleyhodges/sfsymbols/sfSunMax';
import {sfSquareFill} from '@bradleyhodges/sfsymbols/sfSquareFill';
import {sfCircleDottedAndCircle} from '@bradleyhodges/sfsymbols/sfCircleDottedAndCircle';
import {sfRectangle} from '@bradleyhodges/sfsymbols/sfRectangle';
import {sfPersonFill} from '@bradleyhodges/sfsymbols/sfPersonFill';
import {sfSquareGrid3x3MiddleFilled} from '@bradleyhodges/sfsymbols/sfSquareGrid3x3MiddleFilled';
import {sfPlus} from '@bradleyhodges/sfsymbols/sfPlus';
import {sfTrash} from '@bradleyhodges/sfsymbols/sfTrash';
import {sfEraser} from '@bradleyhodges/sfsymbols/sfEraser';
import {sfSpeakerWave2} from '@bradleyhodges/sfsymbols/sfSpeakerWave2';
import {sfSpeakerSlash} from '@bradleyhodges/sfsymbols/sfSpeakerSlash';
import {sfMagnifyingglass} from '@bradleyhodges/sfsymbols/sfMagnifyingglass';
import {sfTextformat} from '@bradleyhodges/sfsymbols/sfTextformat';
import {sfPhoto} from '@bradleyhodges/sfsymbols/sfPhoto';
import {sfDrop} from '@bradleyhodges/sfsymbols/sfDrop';
import {sfSliderHorizontal3} from '@bradleyhodges/sfsymbols/sfSliderHorizontal3';
const icons={previousFrame:sfBackwardFrameFill,nextFrame:sfForwardFrameFill,cut:sfScissors,wave:sfWaveform,link:sfLink,play:sfPlayFill,pause:sfPauseFill,export:sfSquareAndArrowUp,boxes:sfViewfinder,palette:sfPaintpalette,info:sfCircleBadgeQuestionmark,chevron:sfChevronDown,undo:sfArrowUturnBackward,redo:sfArrowUturnForward,check:sfCheckmark,brightness:sfSunMax,noir:sfSquareFill,gaussien:sfCircleDottedAndCircle,pixels:sfSquareGrid3x3MiddleFilled,rectangle:sfRectangle,lisse:sfPersonFill,pixelise:sfSquareGrid3x3MiddleFilled,add:sfPlus,delete:sfTrash,clear:sfEraser,sound:sfSpeakerWave2,mute:sfSpeakerSlash,search:sfMagnifyingglass,text:sfTextformat,image:sfPhoto,watermark:sfDrop,tools:sfSliderHorizontal3};
export type SFName=keyof typeof icons;
export function SFIcon({name,size=16}:{name:SFName;size?:number}){const icon=icons[name];return <svg className="sf-icon" width={size} height={size} viewBox={icon.viewBox} fill="currentColor" aria-hidden="true" data-sf-symbol={icon.sourceName}>{icon.svgPathData.map((path,i)=><path key={i} {...path}/>)}</svg>}

export function CloseGlyph(){return <svg className="close-glyph" width="24" height="24" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="12"/><path d="m8 8 8 8m-8 0 8-8" fill="none" strokeWidth="2" strokeLinecap="round"/></svg>}
