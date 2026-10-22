"""Détection locale de têtes, y compris vue de dos (approximative).

Un ONNX YOLO 'head' pré-entraîné (HollywoodHeads) est prioritaire. Sinon le modèle
COCO détecte des PERSONNES et on masque volontairement une grande région
supérieure. Cette solution de secours n'est PAS un détecteur de têtes fiable.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

HEAD_MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'head_detector.onnx'
Box = tuple[int, int, int, int]


def person_to_head(person: Box, width: int, height: int) -> Box:
    """Région protectrice en haut d'une silhouette visible (face OU dos).

    Couverture volontairement large : lorsque la personne n'est pas entière
    dans le cadre, la localisation de la tête peut être incorrecte.
    """
    x, y, w, h = person
    head_h = min(h * .24, max(h * .18, w * .45))
    head_w = min(w * .70, head_h * 1.05)
    x1 = max(0, round(x + w * .5 - head_w * .5))
    x2 = min(width, round(x + w * .5 + head_w * .5))
    y1 = max(0, round(y - head_h * .08))
    y2 = min(height, round(y + head_h * .92))
    return (x1, y1, max(1,x2-x1), max(1,y2-y1))


class YoloHeadONNX:
    """YOLOv8/v11 ONNX mono-classe, sortie [1,5,N] ou [1,N,5]."""
    def __init__(self, path: Path = HEAD_MODEL_PATH):
        import onnxruntime as ort
        self.path = path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = (['DmlExecutionProvider'] if 'DmlExecutionProvider' in ort.get_available_providers() else []) + ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        self.name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.size = int(shape[-1]) if isinstance(shape[-1],int) else 640
        self.provider = self.session.get_providers()[0]
        self.ort = ort

    def detect(self, frame: np.ndarray, threshold: float = .21) -> list[Box]:
        return [box for box,_ in self.detect_scored(frame,threshold)]

    def detect_scored(self, frame: np.ndarray, threshold: float = .21):
        h,w = frame.shape[:2]
        size = self.size
        scale = min(size / w, size / h)
        nw, nh = max(1,round(w*scale)), max(1,round(h*scale))
        left,top = (size-nw)//2,(size-nh)//2
        canvas = np.full((size,size,3),114,np.uint8)
        canvas[top:top+nh,left:left+nw] = cv2.resize(frame,(nw,nh))
        tensor = cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).transpose(2,0,1)[None].astype('float32')/255.
        try:
            raw = self.session.run(None,{self.name:tensor})[0]
        except Exception:
            if self.provider == 'CPUExecutionProvider':
                raise
            self.session = self.ort.InferenceSession(str(self.path),providers=['CPUExecutionProvider'])
            self.provider = 'CPUExecutionProvider'
            raw = self.session.run(None,{self.name:tensor})[0]
        pred = np.squeeze(raw)
        if pred.ndim != 2:
            raise ValueError('Sortie du modèle head non compatible (attendu [N,5]).')
        if pred.shape[0] == 5:
            pred = pred.T
        if pred.shape[1] != 5:
            raise ValueError(f'Le détecteur de têtes doit être YOLO 1 classe [N,5], reçu {pred.shape}.')
        sel = pred[pred[:,4] >= threshold]
        if len(sel) > 500:
            sel = sel[np.argsort(sel[:,4])[-500:]]
        boxes, scores = [], []
        for cx,cy,bw,bh,confidence in sel:
            x1 = max(0,round((float(cx-bw/2)-left)/scale))
            y1 = max(0,round((float(cy-bh/2)-top)/scale))
            x2 = min(w,round((float(cx+bw/2)-left)/scale))
            y2 = min(h,round((float(cy+bh/2)-top)/scale))
            if x2-x1 >= 4 and y2-y1 >= 4:
                boxes.append((x1,y1,x2-x1,y2-y1))
                scores.append(float(confidence))
        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes,scores,threshold,.45)
        return [(boxes[int(i)],scores[int(i)]) for i in np.asarray(keep).reshape(-1)] if keep is not None else []


class HeadDetector:
    def __init__(self, coco_detector=None):
        self.coco = coco_detector
        self.custom = YoloHeadONNX() if HEAD_MODEL_PATH.is_file() else None
        if self.coco is None and self.custom is None:
            raise FileNotFoundError('Détection tête/dos : lancer INSTALLER.bat pour installer YOLO Head et COCO sans entraîner.')
        self.provider = ('Têtes YOLO Head pré-entraîné ('+self.custom.provider+')' if self.custom else '') + (' + personnes COCO' if coco_detector else '')

    def detect(self, frame: np.ndarray) -> list[Box]:
        h,w = frame.shape[:2]
        heads = self.custom.detect(frame) if self.custom else []
        if self.coco:
            people = self.coco.detect(frame,{0},.22)
            for b in people:
                candidate = person_to_head(b,w,h)
                # Les deux approches se complètent ; les faux positifs peuvent
                # devenir nombreux, volontairement pour privilégier le rappel.
                heads.append(candidate)
        return heads
