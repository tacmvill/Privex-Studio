"""Détecteurs de visages entièrement locaux.

SCRFD (InsightFace, poids NON commerciaux) + YuNet (OpenCV Zoo).
Démarrage interdit quand la détection demandée n'est pas disponible : éviter
les exports trompeusement « anonymisés » sans modèle.
"""
from __future__ import annotations

from pathlib import Path
from .custom_faces import CUSTOM_FACE_PATH, CustomYoloFaceDetector
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRFD_PATH = ROOT / 'models' / 'det_10g.onnx'
YUNET_PATH = ROOT / 'models' / 'face_detection_yunet_2023mar.onnx'
Box = tuple[int, int, int, int]


def _nms(boxes: list[Box], scores: list[float], threshold: float = .38) -> list[Box]:
    if not boxes:
        return []
    kept = cv2.dnn.NMSBoxes([list(b) for b in boxes], scores, .05, threshold)
    if kept is None or len(kept) == 0:
        return []
    return [boxes[int(i)] for i in np.asarray(kept).reshape(-1)]


def _clip(box: tuple[float, float, float, float], width: int, height: int) -> Box | None:
    if not all(np.isfinite(v) for v in box):return None
    x1, y1, x2, y2 = box
    x1, y1 = max(0, round(x1)), max(0, round(y1))
    x2, y2 = min(width, round(x2)), min(height, round(y2))
    return (x1, y1, x2-x1, y2-y1) if x2-x1 >= 3 and y2-y1 >= 3 else None


class SCRFDDetector:
    """Inférence brute du fichier det_10g.onnx. Pas besoin d'InsightFace/PyTorch à l'exécution."""

    def __init__(self, path: Path = SCRFD_PATH):
        if not path.exists():
            raise FileNotFoundError(f'Modèle SCRFD manquant : {path}. Lancez INSTALLER.bat.')
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('ONNX Runtime manquant. Lancez INSTALLER.bat.') from exc
        self.ort = ort
        self.path = path
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider'] if 'DmlExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(str(path), providers=providers, sess_options=options)
        except Exception:
            self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self.name = self.session.get_inputs()[0].name
        inputs = self.session.get_inputs()[0].shape
        self.static_size = (int(inputs[3]), int(inputs[2])) if all(isinstance(v, int) and v > 0 for v in inputs[2:4]) else None
        outputs = len(self.session.get_outputs())
        if outputs not in (6, 9, 10, 15):
            raise ValueError(f'Architecture SCRFD inconnue : {outputs} sorties.')
        self.fmc = 3 if outputs in (6, 9) else 5
        self.strides = [8, 16, 32] if self.fmc == 3 else [8, 16, 32, 64, 128]
        self.anchors = 2 if self.fmc == 3 else 1
        self.provider = self.session.get_providers()[0]
        self._center_cache: dict[tuple[int,int,int], np.ndarray] = {}

    def _centers(self, width: int, height: int, stride: int) -> np.ndarray:
        key = width, height, stride
        if key not in self._center_cache:
            xx, yy = np.meshgrid(np.arange(width // stride), np.arange(height // stride))
            centers = np.stack([xx, yy], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            self._center_cache[key] = np.repeat(centers, self.anchors, axis=0)
        return self._center_cache[key]

    def _run(self, blob: np.ndarray):
        shape=tuple(blob.shape)
        if not hasattr(self,'_shape_sessions'):self._shape_sessions={}
        session=self._shape_sessions.get(shape,self.session)
        if shape[-1]!=640 and shape not in self._shape_sessions:
            opts=self.ort.SessionOptions();opts.enable_mem_pattern=False;opts.execution_mode=self.ort.ExecutionMode.ORT_SEQUENTIAL;opts.intra_op_num_threads=4;opts.logid=f'SCRFD-{shape[-1]}'
            # SCRFD's dynamic Reshape at 960 is incompatible with the tested DML runtime.
            # Keep the precise model on CPU at this size and preserve GPU at 640.
            self._shape_sessions[shape]=self.ort.InferenceSession(str(self.path),sess_options=opts,providers=['CPUExecutionProvider']);session=self._shape_sessions[shape]
            if self.session.get_providers()[0]=='DmlExecutionProvider':__import__('performance').note_model(self.path.name,'SCRFD 960: CPU (Reshape incompatible DirectML); SCRFD 640 remains on GPU.')
        try:return session.run(None,{self.name:blob})
        except Exception:
            if session.get_providers()[0]=='CPUExecutionProvider':raise
            self._shape_sessions[shape]=self.ort.InferenceSession(str(self.path),providers=['CPUExecutionProvider'])
            return self._shape_sessions[shape].run(None,{self.name:blob})

    def detect(self, frame: np.ndarray, size: int = 640, threshold: float = .27,
               enhance: bool = False) -> list[tuple[Box, float]]:
        h, w = frame.shape[:2]
        iw, ih = self.static_size if self.static_size else (size, size)
        scale = min(iw / w, ih / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        image = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        if enhance:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(lab[:,:,0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        canvas = np.zeros((ih, iw, 3), np.uint8)
        canvas[:nh, :nw] = image
        blob = cv2.dnn.blobFromImage(canvas, 1/128.0, (iw,ih), (127.5,127.5,127.5), swapRB=True)
        outputs = self._run(blob)
        found = []
        for n, stride in enumerate(self.strides):
            scores = np.asarray(outputs[n]).reshape(-1)
            distances = np.asarray(outputs[n+self.fmc]).reshape(-1, 4) * stride
            centers = self._centers(iw, ih, stride)
            if len(centers) != len(scores) or len(scores) != len(distances):
                raise ValueError(f'SCRFD sortie incohérente, stride {stride}: scores={len(scores)} ancres={len(centers)}')
            indices = np.flatnonzero(scores >= threshold)
            for j in indices:
                cx, cy = centers[j]
                l, t, r, b = distances[j]
                box = _clip(((cx-l)/scale, (cy-t)/scale, (cx+r)/scale, (cy+b)/scale), w, h)
                if box:
                    found.append((box, float(scores[j])))
        boxes = [b for b, _ in found]
        scores = [s for _, s in found]
        kept = _nms(boxes, scores)
        return [(box, max(s for b,s in found if b == box)) for box in kept]


class YuNetDetector:
    def __init__(self, path: Path = YUNET_PATH):
        if not path.exists():
            raise FileNotFoundError(f'Modèle YuNet manquant : {path}. Lancez INSTALLER.bat.')
        self.model = cv2.FaceDetectorYN.create(str(path), '', (320,320), score_threshold=.53,
                                               nms_threshold=.35, top_k=5000)

    def detect(self, frame: np.ndarray, enhanced: bool = False) -> list[tuple[Box,float]]:
        h, w = frame.shape[:2]
        scale = min(2., 1280 / max(w, h))  # petits visages des vidéos compressées
        image = cv2.resize(frame, (round(w*scale),round(h*scale))) if scale > 1 else frame
        if enhanced:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(lab[:,:,0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        ih, iw = image.shape[:2]
        self.model.setInputSize((iw, ih))
        _, faces = self.model.detect(image)
        if faces is None:
            return []
        result = []
        for f in faces:
            box = _clip((f[0]/scale, f[1]/scale, (f[0]+f[2])/scale, (f[1]+f[3])/scale), w, h)
            if box:
                result.append((box,float(f[-1])))
        return result


def tile_offsets(length: int, tile: int, overlap: float = .32) -> list[int]:
    """Positions déterministes, avec la dernière tuile collée au bord.

    Le recouvrement est important : un visage à cheval entre deux tuiles
    ne doit pas disparaître. En-dessous de la taille de la tuile, une seule
    extraction suffit.
    """
    if length <= tile:
        return [0]
    stride = max(1, round(tile * (1.0 - overlap)))
    offsets = list(range(0, max(1, length - tile + 1), stride))
    if offsets[-1] != length - tile:
        offsets.append(length - tile)
    return offsets


def _translate_box(box: Box, ox: int, oy: int, w: int, h: int) -> Box | None:
    x, y, bw, bh = box
    return _clip((x + ox, y + oy, x + bw + ox, y + bh + oy), w, h)


class HighRecallFaceDetector:
    """Mode extrême : recherche multi-échelle + tuiles + miroir.

    Un rappel élevé implique beaucoup de faux positifs et n'est PAS une
    certification d'anonymisation. Les poids SCRFD sont soumis à licence.
    """

    def __init__(self, require_scrfd: bool = True):
        self.scrfd = SCRFDDetector() if require_scrfd else None
        self.yunet = YuNetDetector()
        self.custom = CustomYoloFaceDetector() if require_scrfd and CUSTOM_FACE_PATH.is_file() else None
        self.provider = (self.scrfd.provider if self.scrfd else 'OpenCV CPU (YuNet)') + (' + custom YOLO-face (' + self.custom.provider + ')' if self.custom else '')

    def _extreme_tiles(self, frame: np.ndarray) -> list[tuple[Box, float]]:
        """Détecte dans des sous-images natives pour agrandir les petites têtes.

        L'image complète réduite à 640px peut faire disparaître les visages
        de 5-15px ; une tuile conserve la résolution relative, même lorsque
        le modèle ONNX est verrouillé sur des entrées 640x640.
        """
        if self.scrfd is None:
            return []
        h, w = frame.shape[:2]
        side = max(96, min(768, round(min(h, w) * .72)))
        side = min(side, w, h)
        xs = tile_offsets(w, side)
        ys = tile_offsets(h, side)
        positions = [(x, y) for y in ys for x in xs]
        # Toutes les tuiles sont examinées, même pour des vidéos très larges.
        # Le temps de traitement peut donc devenir considérable.
        found: list[tuple[Box, float]] = []
        for x, y in positions:
            crop = frame[y:y + side, x:x + side]
            # Pas de traitement de l'image plein cadre une deuxième fois.
            if x == 0 and y == 0 and side == w and side == h:
                continue
            for b, score in self.scrfd.detect(crop, size=640, threshold=.15):
                translated = _translate_box(b, x, y, w, h)
                if translated:
                    found.append((translated, score))
        return found

    def detect(self, frame: np.ndarray, quality: str = 'extreme') -> list[Box]:
        if quality not in ('base', 'regional', 'standard', 'precise', 'extreme'):
            raise ValueError(f'Mode de détection de visages inconnu : {quality}')
        proposals: list[tuple[Box, float]] = []
        if self.scrfd:
            proposals += self.scrfd.detect(frame, size=640,
                                           threshold=.32 if quality in ('base','standard') else .28)
            if quality in ('regional','precise', 'extreme'):
                if self.scrfd.static_size is None and quality!='regional':
                    proposals += self.scrfd.detect(frame, size=960, threshold=.19)
                proposals += self.scrfd.detect(frame, size=640,
                                               threshold=.18 if quality == 'extreme' else .27,
                                               enhance=True)
            if quality == 'extreme':
                proposals += self._extreme_tiles(frame)
                flipped = cv2.flip(frame, 1)
                for b, score in self.scrfd.detect(flipped, size=640, threshold=.17):
                    x, y, bw, bh = b
                    proposals.append(((frame.shape[1] - x - bw, y, bw, bh), score))
        if quality in ('regional','precise', 'extreme') and self.custom is not None:
            proposals += self.custom.detect(frame, threshold=.17 if quality == 'extreme' else .22)
        # A lone low-confidence YuNet response on a hand/sign must not override
        # SCRFD's negative evidence. Keep independent high-confidence faces and
        # corroborated low-confidence faces; native person crops recover small ones.
        yunet = self.yunet.detect(frame) if quality!='standard' or self.scrfd is None else []
        if self.scrfd:
            from head_fusion import same_head
            yunet = [(b,s) for b,s in yunet if s>=.85 or any(same_head(p,b) for p,_ in proposals)]
        proposals += yunet
        if quality in ('regional','precise', 'extreme'):
            proposals += [(b,s) for b,s in self.yunet.detect(frame, enhanced=True) if s>=.85 or any(__import__('head_fusion').same_head(p,b) for p,_ in proposals)]
        # Contrairement au NMS classique, ne supprime que les quasi-doublons
        # afin de ne pas effacer deux visages rapprochés dans une foule.
        self.last_scores={}
        for box,score in proposals:self.last_scores[box]=max(score,self.last_scores.get(box,0.))
        return _nms([b for b, s in proposals], [s for b, s in proposals],
                    .82 if quality == 'extreme' else .55 if quality=='base' else .4)

    def detect_scored(self,frame,quality='extreme'):
        return [(box,float(self.last_scores[box])) for box in self.detect(frame,quality)]
