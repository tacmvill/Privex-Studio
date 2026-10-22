/** Stable per-subject color, independent of whether masking has been applied. */
export function subjectColor(id:string):string{let hash=2166136261;for(const c of id.replace(/^m-da/,'a'))hash=Math.imul(hash^c.charCodeAt(0),16777619);const value=hash>>>0;return `hsl(${value%360} ${65+(value>>>12)%20}% ${56+(value>>>20)%10}%)`}
