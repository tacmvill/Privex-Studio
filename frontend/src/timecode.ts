/** Shared frame display for monitor, timeline and hover previews (MM:SS:FF). */
export function frameTimecode(seconds:number,rate=25):string{
 const t=Number.isFinite(seconds)?Math.max(0,seconds):0;
 const fps=Number.isFinite(rate)&&rate>0?rate:25;
 const whole=Math.floor(t),frame=Math.min(Math.ceil(fps)-1,Math.floor((t-whole)*fps+1e-6));
 return `${String(Math.floor(whole/60)).padStart(2,'0')}:${String(whole%60).padStart(2,'0')}:${String(frame).padStart(2,'0')}`;
}
