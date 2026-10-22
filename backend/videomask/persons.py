"""Détéction de personnes COCO et segmentation ONNX optionnelle, totalement locale.

La voie "corps entier" conserve des boîtes englobantes, qui couvrent aussi
les cheveux / bras / pieds lorsqu'un modèle COCO reconnaît la personne.
Le mode silhouette est uniquement esthétique et susceptible d'omettre les
bords. Un défaut de segmentation déclenche un REPLI en boîte pleine.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]
PERSON_SEG_PATH = Path(__file__).resolve().parents[1] / 'models' / 'yolo11n-seg.onnx'


@dataclass
class Instance:
    box: Box
    mask: np.ndarray  # uint8 0 ou 255, dimensions de l'image
    score: float
    class_id: int = 0


def iou(a: Box, b: Box) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0,xx2-xx1)*max(0,yy2-yy1)
    area = a[2]*a[3] + b[2]*b[3] - inter
    return inter / area if area else 0.


class PersonSegmentationONNX:
    """YOLO11n-seg, sorties raw [1,116,N] et [1,32,H,W].

    Le modèle doit avoir 80 classes COCO et 32 coefficients de masque.
    Aucun import d'Ultralytics ni téléchargement à l'exécution.
    """
    def __init__(self, path: Path = PERSON_SEG_PATH):
        import onnxruntime as ort
        self.ort = ort
        self.path = Path(path)
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = (['DmlExecutionProvider'] if 'DmlExecutionProvider' in ort.get_available_providers() else []) + ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(str(self.path), sess_options=opts, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(str(self.path), sess_options=opts, providers=['CPUExecutionProvider'])
        self.provider = self.session.get_providers()[0]
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else 640

    def _run(self, tensor: np.ndarray):
        try:
            return self.session.run(None, {self.input_name: tensor})
        except Exception:
            if self.provider == 'CPUExecutionProvider':
                raise
            self.session = self.ort.InferenceSession(str(self.path), providers=['CPUExecutionProvider'])
            self.provider = 'CPUExecutionProvider'
            return self.session.run(None, {self.input_name: tensor})

    def predict(self, frame: np.ndarray, threshold: float = .18, classes: set[int] | None = None) -> list[Instance]:
        h,w = frame.shape[:2]
        s = self.size
        scale = min(s/w,s/h)
        nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
        l,t = (s-nw)//2,(s-nh)//2
        canvas = np.full((s,s,3),114,np.uint8)
        canvas[t:t+nh,l:l+nw] = cv2.resize(frame,(nw,nh))
        tensor = cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).transpose(2,0,1)[None].astype(np.float32)/255.
        outputs = self._run(tensor)
        if len(outputs) != 2:
            raise ValueError('ONNX seg : deux sorties YOLO (détections + prototypes) requises.')
        proto = np.squeeze(outputs[1])
        if proto.ndim != 3 or proto.shape[0] != 32:
            raise ValueError(f'Prototypes YOLO seg invalides : {proto.shape}')
        pred = np.asarray(outputs[0])
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.ndim != 2:
            raise ValueError(f'Détections YOLO seg invalides : {pred.shape}')
        if pred.shape[0] == 116:
            pred = pred.T
        if pred.shape[1] != 116:
            raise ValueError(f'YOLO segmentation COCO 80 classes attendu [N,116] : {pred.shape}')
        # Classe COCO 0 = person. Aucun objet non-personne n'est masqué.
        scores = np.max(pred[:,4:84],axis=1)
        bestcls = np.argmax(pred[:,4:84],axis=1)
        valid = np.flatnonzero(np.isin(bestcls,list(classes if classes is not None else {0})) & (scores >= threshold))
        if not len(valid):
            return []
        valid = valid[np.argsort(scores[valid])[::-1][:150]]
        bb, confidences, coeffs, class_ids = [], [], [], []
        for i in valid:
            cx,cy,bw,bh = [float(v) for v in pred[i,:4]]
            x1 = max(0,round((cx-bw/2-l)/scale)); y1=max(0,round((cy-bh/2-t)/scale))
            x2 = min(w,round((cx+bw/2-l)/scale)); y2=min(h,round((cy+bh/2-t)/scale))
            if x2-x1 < 4 or y2-y1 < 4:
                continue
            bb.append((x1,y1,x2-x1,y2-y1))
            confidences.append(float(scores[i]))
            coeffs.append(pred[i,84:116]);class_ids.append(int(bestcls[i]))
        if not bb:
            return []
        kept = cv2.dnn.NMSBoxesBatched(bb,confidences,class_ids,threshold,.5)
        if kept is None or len(kept)==0:
            return []
        # Le masque est calculé seulement pour les personnes retenues par NMS.
        instances = []
        ph,pw = proto.shape[1:]
        prototypes = proto.reshape(32,-1)
        for idx in np.asarray(kept).reshape(-1):
            idx = int(idx)
            x,y,bw,bh=bb[idx]
            logits = (coeffs[idx] @ prototypes).reshape(ph,pw)
            m = (logits > 0).astype(np.uint8) * 255  # sigmoid(logits) > .5
            # Retour depuis le letterbox vers les dimensions exactes de la source.
            crop_x1=max(0,round(l*pw/s)); crop_y1=max(0,round(t*ph/s))
            crop_x2=min(pw,round((l+nw)*pw/s)); crop_y2=min(ph,round((t+nh)*ph/s))
            if crop_x2<=crop_x1 or crop_y2<=crop_y1:
                continue
            m = cv2.resize(m[crop_y1:crop_y2,crop_x1:crop_x2],(w,h),interpolation=cv2.INTER_LINEAR)
            m = (m>127).astype(np.uint8)*255
            # Empêche la segmentation de déborder très loin de sa boîte.
            roi = np.zeros_like(m)
            roi[y:y+bh,x:x+bw] = m[y:y+bh,x:x+bw]
            instances.append(Instance(bb[idx],roi,confidences[idx],class_ids[idx]))
        return instances


class PersonDetector:
    """Privilégie le rappel : COCO boîte + segmentation facultative."""
    def __init__(self,coco=None,segmentation: bool=False):
        self.coco = coco
        self.segment = PersonSegmentationONNX() if segmentation else None
        if self.coco is None and self.segment is None:
            raise FileNotFoundError('Modèle de personnes manquant : lancer INSTALLER.bat.')
        self.provider = ('Personnes COCO '+getattr(coco,'provider','') if coco else '') + (
            (' + silhouettes ONNX '+self.segment.provider) if self.segment else '')

    def detect(self, frame: np.ndarray) -> list[Box]:
        boxes = list(self.coco.detect(frame,{0},.18)) if self.coco else []
        # Union COCO + segmentation : si un seul modèle voit la personne,
        # celle-ci reste dans le plan temporel (rappel > précision).
        if self.segment:
            for instance in self.segment.predict(frame):
                if not any(iou(instance.box, b) >= .55 for b in boxes):
                    boxes.append(instance.box)
        return boxes

    def masks(self, frame: np.ndarray) -> list[Instance]:
        return self.segment.predict(frame) if self.segment else []
