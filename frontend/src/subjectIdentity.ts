export function relatedSubjectIds(id:string,rows:{id:string;reference_ids?:string[]}[]){
 const key=(s:string)=>s.startsWith('m-da')?s.slice(3):s;
 const found=new Set([key(id)]);let changed=true;
 while(changed){changed=false;for(const row of rows){const ids=[key(row.id),...(row.reference_ids||[]).map(key)];if(ids.some(v=>found.has(v)))for(const v of ids)if(!found.has(v)){found.add(v);changed=true}}}
 return [...found];
}
