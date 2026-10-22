"""Détecteur YOLO-face mono-classe entraîné sur WIDER FACE.

Prend uniquement les exports YOLO11 / Ultralytics ONNX sans NMS intégrée
avec sortie (1,5,N) ou (1,N,5) = (cx,cy,w,h,score). Le modèle ne doit pas
être remplacé par un modèle SCRFD/RetinaFace/COCO standard.
"""
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CUSTOM_FACE_PATH=ROOT/'models'/'yolo_face_custom.onnx'

class CustomYoloFaceDetector:
    def __init__(self,path:Path=CUSTOM_FACE_PATH):
        import onnxruntime as ort
        self.ort=ort
        self.path=Path(path)
        opts=ort.SessionOptions()
        opts.enable_mem_pattern=False
        opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        providers=['DmlExecutionProvider','CPUExecutionProvider'] if 'DmlExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        try:
            self.session=ort.InferenceSession(str(self.path),sess_options=opts,providers=providers)
        except Exception:
            self.session=ort.InferenceSession(str(self.path),providers=['CPUExecutionProvider'])
        self.provider=self.session.get_providers()[0]
        self.input_name=self.session.get_inputs()[0].name
        dims=self.session.get_inputs()[0].shape
        self.h=int(dims[2]) if isinstance(dims[2],int) and dims[2]>0 else 640
        self.w=int(dims[3]) if isinstance(dims[3],int) and dims[3]>0 else 640
        if self.h%32 or self.w%32: raise ValueError('Entrée YOLO personnalisée non conforme')

    def detect(self,frame:np.ndarray,threshold:float=.22):
        height,width=frame.shape[:2]
        scale=min(self.w/width,self.h/height)
        nw=max(1,round(width*scale)); nh=max(1,round(height*scale))
        resized=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_LINEAR)
        left=(self.w-nw)//2;top=(self.h-nh)//2
        canvas=np.full((self.h,self.w,3),114,np.uint8)
        canvas[top:top+nh,left:left+nw]=resized
        tensor=np.transpose(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB),(2,0,1))[None].astype(np.float32)/255.
        try:
            pred=self.session.run(None,{self.input_name:tensor})[0]
        except Exception:
            if self.provider=='CPUExecutionProvider': raise
            self.session=self.ort.InferenceSession(str(self.path),providers=['CPUExecutionProvider'])
            self.provider='CPUExecutionProvider'
            self.input_name=self.session.get_inputs()[0].name
            pred=self.session.run(None,{self.input_name:tensor})[0]
        pred=np.asarray(pred)
        if pred.ndim==3 and pred.shape[0]==1:
            pred=pred[0]
        if pred.ndim!=2:
            raise ValueError(f'ONNX personnalisé : sortie invalide {pred.shape} ; exporter avec nms=False')
        if pred.shape[0]==5: pred=pred.T
        if pred.shape[1]!=5:
            raise ValueError(f'ONNX personnalisé : 1 classe face nécessaire, sortie {pred.shape}')
        indices=np.flatnonzero(pred[:,4]>=threshold)
        if not len(indices):return []
        indices=indices[np.argsort(pred[indices,4])[::-1][:1000]]
        boxes=[]; scores=[]
        for i in indices:
            cx,cy,bw,bh=[float(v) for v in pred[i,:4]]
            x1=max(0,round((cx-bw/2-left)/scale));y1=max(0,round((cy-bh/2-top)/scale))
            x2=min(width,round((cx+bw/2-left)/scale));y2=min(height,round((cy+bh/2-top)/scale))
            if x2-x1>=3 and y2-y1>=3:
                boxes.append((x1,y1,x2-x1,y2-y1));scores.append(float(pred[i,4]))
        if not boxes:return []
        kept=cv2.dnn.NMSBoxes(boxes,scores,threshold,.4)
        if kept is None or not len(kept):return []
        return [(boxes[int(i)],scores[int(i)]) for i in np.asarray(kept).reshape(-1)]
