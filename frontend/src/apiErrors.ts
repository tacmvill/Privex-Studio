const fieldNames:Record<string,string>={references:'sujets détectés',selected_ids:'sujets sélectionnés',custom:'suivi personnalisé',box:'zone du sujet',anchors:'points de suivi',grounding_prompts:'descriptions des sujets',analysis_ranges:'passages à analyser',temporal_decisions:'décisions de suivi',text_overlays:'textes du montage',image_overlays:'images superposées',profile:'puissance du moteur',effect:'effet du masque',shape:'contour du masque',margin:'marge de sécurité',size:'taille du masque',tile_size:'taille des tuiles',confidence:'seuil de détection',frame_step:'intervalle entre les images',classes:'catégories de sujets',segmentation:'segmentation',tracker:'suivi',mosaic_color:'couleur de la mosaïque'};

export function formatApiError(detail:unknown):string{
 if(typeof detail==='string')return detail;
 if(!Array.isArray(detail)||!detail.length)return 'Le moteur local n’a pas pu utiliser ces réglages.';
 const messages=detail.map(value=>{
  const issue=value&&typeof value==='object'?value as {loc?:unknown[];type?:string;msg?:string;ctx?:{max_length?:number}}:{};
  const fields=Array.isArray(issue.loc)?issue.loc.filter((v):v is string=>typeof v==='string'&&v!=='body'&&v!=='settings'):[];
  const field=[...fields].reverse().find(v=>fieldNames[v]);
  const label=field?fieldNames[field]:'paramètre envoyé';
  if(issue.type==='too_long'&&typeof issue.ctx?.max_length==='number')return `La liste « ${label} » dépasse la limite de ${issue.ctx.max_length} éléments.`;
  if(issue.type==='missing')return `Le réglage « ${label} » est manquant.`;
  if(issue.type==='value_error'&&issue.msg?.startsWith('Value error, '))return `${label} : ${issue.msg.slice(13)}`;
  return `Vérifiez la valeur du réglage « ${label} ».`;
 });
 return [...new Set(messages)].slice(0,2).join(' ');
}
