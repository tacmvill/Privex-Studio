"""Local fixed-prompt YOLOE + PP-OCRv4. Open vocabulary labels require review."""
from pathlib import Path
import json,re,unicodedata
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
EXTENDED_NAMES=json.loads((ROOT/'models/open-vocabulary.json').read_text(encoding='utf-8'))['labels']
BRANDS=['Nike','Apple','Orange','Renault','Ferrari','Nutella','Cristaline','Acer','Samsung','Ralph Lauren','Lacoste']

class ExtendedDetector:
 def __init__(self,names,ocr=False,brands=False):
  self.recognize_text=ocr;self.brands=brands;self.names=set(names);self.session=None;self.ocr=None
  if names:
   import onnxruntime as ort
   options=ort.SessionOptions();options.intra_op_num_threads=4
   self.session=ort.InferenceSession(str(ROOT/'models/open-vocabulary.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
  if ocr or brands:
   from rapidocr_onnxruntime import RapidOCR
   self.ocr=RapidOCR(intra_op_num_threads=4,inter_op_num_threads=1)
 def detect(self,frame,threshold):
  h,w=frame.shape[:2];result=[]
  if self.session:
   scale=min(640/w,640/h);nw,nh=round(w*scale),round(h*scale);left,top=(640-nw)//2,(640-nh)//2;canvas=np.full((640,640,3),114,np.uint8);canvas[top:top+nh,left:left+nw]=cv2.resize(frame,(nw,nh))
   tensor=np.ascontiguousarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).transpose(2,0,1)[None],dtype=np.float32)/255
   pred=self.session.run(None,{self.session.get_inputs()[0].name:tensor})[0][0].T;n=len(EXTENDED_NAMES);ids=np.argmax(pred[:,4:4+n],axis=1);scores=pred[np.arange(len(pred)),ids+4]
   for cls,name in enumerate(EXTENDED_NAMES):
    if name not in self.names:continue
    boxes=[];conf=[]
    for index in np.flatnonzero((ids==cls)&(scores>=max(.18,threshold))):
     cx,cy,bw,bh=pred[index,:4];x=max(0,round((cx-bw/2-left)/scale));y=max(0,round((cy-bh/2-top)/scale));bw=min(w-x,round(bw/scale));bh=min(h-y,round(bh/scale))
     if bw>4 and bh>4:boxes.append([x,y,bw,bh]);conf.append(float(scores[index]))
    keep=cv2.dnn.NMSBoxes(boxes,conf,max(.18,threshold),.45) if boxes else []
    for index in np.asarray(keep).reshape(-1):result.append({'box':boxes[int(index)],'class':name,'class_id':200+cls,'confidence':conf[int(index)],'review_required':True,'method':'YOLOE · proposition à vérifier'})
  if self.ocr:
   # Read the actual text; brand names do not imply visual logo recognition.
   rows,_=self.ocr(frame)
   for polygon,text,score in rows or []:
    if score<max(.5,threshold):continue
    x,y,bw,bh=cv2.boundingRect(np.array(polygon,dtype=np.int32));name='texte';text_lower=unicodedata.normalize('NFKD',text.casefold())
    for brand in BRANDS:
     if re.search(r'\b'+re.escape(brand.casefold())+r'\b',text_lower):name='marque '+brand;break
    if name.startswith('marque ') and not self.brands:name='texte'
    if not self.recognize_text and not name.startswith('marque '):continue
    if self.recognize_text and re.search(r'\b\d{1,2}\s*[:h]\s*\d{2}\b',text_lower):name='heure lisible'
    elif self.recognize_text and any(t in text_lower for t in ['rue ','avenue','boulevard','place ','paris']):name='texte de panneau'
    result.append({'box':[x,y,bw,bh],'class':name,'class_id':300+(['texte','heure lisible','texte de panneau',*['marque '+b for b in BRANDS]].index(name)),'confidence':float(score),'text':text,'method':'PP-OCRv4'})
  return result
