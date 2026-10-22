"""Moteur vidéo local : OpenCV, option ONNX Runtime (CPU ou DirectML).

Aucun appel réseau. Les masques ne suppriment ni n'interpolent d'images.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .faces import HighRecallFaceDetector, SCRFD_PATH, YUNET_PATH
from .heads import HeadDetector, HEAD_MODEL_PATH
from .persons import PersonDetector, PERSON_SEG_PATH, iou
from .contours import tiled_silhouette, grabcut_selected
from .custom_tracking import CustomObject

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolo11n.onnx"

# Indices COCO80. Un bureau n'est PAS une classe COCO : la classe 60
# représente « dining table ». Les fenêtres et les documents n'y figurent pas.
ANIMALS = frozenset(range(14, 24))
SCREENS = frozenset({62, 63, 67})  # tv, laptop, cell phone
TABLES = frozenset({60})  # dining table (approximation, pas desk)
Box = tuple[int, int, int, int]


@dataclass
class Settings:
    faces: bool = True
    heads: bool = False  # Têtes complètes, modèle HollywoodHeads pré-entraîné.
    persons: bool = False  # Corps entiers, toutes orientations et cadrages.
    body_padding: float = 0.10  # Marge autour de la silhouette/boîte.
    body_silhouette: bool = False  # Compatibilité V6.
    body_shape: Literal["box", "blocks", "silhouette", "auto"] = "box"
    block_size: int = 14  # grille rectangulaire près du contour
    head_padding: float = 0.55  # expansion du visage et de la tête autour des cheveux
    head_shape: Literal["box", "blocks", "smooth"] = "box"
    fail_closed: bool = False  # image NOIRE si aucune tête détectée (option stricte)
    animals: bool = False
    screens: bool = False
    tables: bool = False
    color: bool = False
    shapes: bool = False
    color_bgr: tuple[int, int, int] = (0, 0, 255)
    hue_tolerance: int = 10
    mode: Literal["black", "blur", "pixel"] = "black"
    inference_stride: int = 4
    face_quality: Literal["extreme", "precise", "standard"] = "extreme"
    confidence: float = 0.35
    regions: list[Box] = field(default_factory=list)
    custom_objects: list[CustomObject] = field(default_factory=list)
    only_selected_subjects: bool = False  # si vrai, seuls les sujets cliqués sont masqués

    @property
    def yolo_classes(self) -> set[int]:
        result: set[int] = set()
        if self.animals:
            result.update(ANIMALS)
        if self.screens:
            result.update(SCREENS)
        if self.tables:
            result.update(TABLES)
        return result


class YoloDetector:
    """YOLO11n ONNX brut (84 x 8400) : aucune dépendance ultralytics à l'exécution."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Modèle IA absent : {path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Installer onnxruntime-directml ou onnxruntime") from exc

        self.ort = ort
        self.path = path
        # Cache un calcul ONNX par image, partagé entre mode tête/personne/objet.
        self._cached_frame = None
        self._cached_prediction = None
        available = ort.get_available_providers()
        providers = (["DmlExecutionProvider"] if "DmlExecutionProvider" in available else [])
        providers.append("CPUExecutionProvider")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.enable_mem_pattern = False  # requis pour DML
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        except Exception:
            # Une carte graphique/pilote incompatible ne bloque pas le CPU.
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.size = int(shape[-1]) if isinstance(shape[-1], int) else 640
        if self.size != 640:
            raise ValueError("Cette V1 attend un modèle YOLO11n ONNX avec imgsz=640.")

    def detect(self, frame: np.ndarray, classes: set[int], threshold: float) -> list[Box]:
        if not classes:
            return []
        h, w = frame.shape[:2]
        size = self.size
        scale = min(size / w, size / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (size - nw) // 2, (size - nh) // 2
        if frame is getattr(self, '_cached_frame', None) and getattr(self, '_cached_prediction', None) is not None:
            pred = self._cached_prediction
        else:
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((size, size, 3), 114, dtype=np.uint8)
            canvas[top:top + nh, left:left + nw] = resized
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            tensor = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0
            try:
                raw = self.session.run(None, {self.input_name: tensor})[0]
            except Exception:
                if self.provider == "CPUExecutionProvider":
                    raise
                # DirectML peut échouer à l'exécution malgré un chargement valide.
                self.session = self.ort.InferenceSession(
                    str(self.path), providers=["CPUExecutionProvider"])
                self.provider = "CPUExecutionProvider"
                self.input_name = self.session.get_inputs()[0].name
                raw = self.session.run(None, {self.input_name: tensor})[0]
            pred = np.squeeze(raw)
            self._cached_frame = frame
            self._cached_prediction = pred
        if pred.ndim != 2:
            raise ValueError(f"Sortie YOLO non reconnue : {pred.shape}")
        if pred.shape[0] == 84:
            pred = pred.T
        if pred.shape[1] != 84:
            raise ValueError(f"Sortie attendue [N,84], reçue {pred.shape}")
        cls = np.argmax(pred[:, 4:], axis=1)
        scores = pred[np.arange(len(pred)), cls + 4]
        wanted = np.isin(cls, list(classes)) & (scores >= threshold)
        ix = np.flatnonzero(wanted)
        if len(ix) == 0:
            return []
        ix = ix[np.argsort(scores[ix])[::-1][:250]]

        xywh: list[list[int]] = []
        confidences: list[float] = []
        for i in ix:
            cx, cy, bw, bh = pred[i, :4]
            x1 = round((float(cx) - float(bw) / 2 - left) / scale)
            y1 = round((float(cy) - float(bh) / 2 - top) / scale)
            x2 = round((float(cx) + float(bw) / 2 - left) / scale)
            y2 = round((float(cy) + float(bh) / 2 - top) / scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 > 3 and y2 - y1 > 3:
                xywh.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(scores[i]))
        if not xywh:
            return []
        kept = cv2.dnn.NMSBoxes(xywh, confidences, threshold, 0.45)
        if kept is None or len(kept) == 0:
            return []
        return [tuple(xywh[int(i)]) for i in np.asarray(kept).reshape(-1)]


def _clip_box(box: Box, w: int, h: int, padding: float = 0.0) -> Box | None:
    x, y, bw, bh = box
    px, py = round(bw * padding), round(bh * padding)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(w, x + bw + px)
    y2 = min(h, y + bh + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _mask_area(roi: np.ndarray, mode: str) -> np.ndarray:
    if mode == "black":
        return np.zeros_like(roi)
    h, w = roi.shape[:2]
    if mode == "pixel":
        # Blocs volontairement visibles, dimensions proportionnelles.
        small_w = max(1, w // 16)
        small_h = max(1, h // 16)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    sigma = max(18.0, min(75.0, min(w, h) / 2.0))
    return cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _expanded_head_box(box: Box, padding: float) -> Box:
    x, y, bw, bh = box
    padding = min(1.5, max(0., float(padding)))
    left = round(bw * padding)
    top = round(bh * (padding + .28))
    bottom = round(bh * (padding * .70))
    return (x-left, y-top, bw+2*left, bh+top+bottom)


def _head_mask(shape: tuple[int, int], box: Box, padding: float, block_size: int = 14,
               pixelated: bool = True) -> np.ndarray:
    h, w = shape
    expanded = _clip_box(_expanded_head_box(box, padding), w, h)
    mask = np.zeros((h, w), np.uint8)
    if expanded is None:
        return mask
    x, y, bw, bh = expanded
    cx, cy = x + bw / 2.0, y + bh / 2.0
    axes = (max(3, round(bw * 0.48)), max(3, round(bh * 0.52)))
    cv2.ellipse(mask, (round(cx), round(cy)), axes, 0, 0, 360, 255, thickness=-1)
    if pixelated:
        return tiled_silhouette(mask, block=max(4, block_size), dilation=max(1, block_size // 5))
    return mask


def _apply_head_box(frame: np.ndarray, box: Box, mode: str, padding: float,
                    head_shape: str = "box", block_size: int = 14) -> None:
    """Couverture plus haute que large pour masquer cheveux, front, oreilles, nuque.

    `head_shape='box'` applique une zone rectangulaire conservatrice.
    `head_shape='blocks'` approxime la tête par un contour en petits carrés.
    """
    if head_shape in ('blocks', 'smooth'):
        _apply_binary_mask(frame, _head_mask(frame.shape[:2], box, padding, block_size,
                                             pixelated=(head_shape == 'blocks')), mode)
    else:
        _apply_box(frame, _expanded_head_box(box, padding), mode, pad=0)


def _apply_box(frame: np.ndarray, box: Box, mode: str, pad: float = 0.0) -> None:
    h, w = frame.shape[:2]
    clipped = _clip_box(box, w, h, pad)
    if clipped is None:
        return
    x, y, bw, bh = clipped
    roi = frame[y:y + bh, x:x + bw]
    frame[y:y + bh, x:x + bw] = _mask_area(roi, mode)


def _apply_binary_mask(frame: np.ndarray, mask: np.ndarray, mode: str) -> None:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return
    x, y, w, h = cv2.boundingRect(coords)
    roi = frame[y:y+h, x:x+w]
    patch = _mask_area(roi, mode)
    local = mask[y:y+h, x:x+w] > 0
    roi[local] = patch[local]


def color_mask(frame: np.ndarray, bgr: tuple[int, int, int], tolerance: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    target = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    hue = int(target[0])
    sat, val = int(target[1]), int(target[2])
    # Les gris/blancs/noirs exigent une logique différente des couleurs saturées.
    if sat < 35:
        result = ((hsv[:, :, 1] < 50) & (np.abs(hsv[:, :, 2].astype(np.int16) - val) < 35))
    else:
        diff = np.abs(hsv[:, :, 0].astype(np.int16) - hue)
        circular_hue_dist = np.minimum(diff, 180 - diff)
        result = ((circular_hue_dist <= tolerance)
                  & (hsv[:, :, 1] >= max(40, sat - 65))
                  & (hsv[:, :, 2] >= max(30, val - 85)))
    mask = (result.astype(np.uint8) * 255)
    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def shape_mask(frame: np.ndarray) -> np.ndarray:
    """Heuristique géométrique, pas une IA sémantique."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros((h, w), np.uint8)
    area = h * w
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if not (area * 0.002 <= a <= area * 0.5):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        n = len(approx)
        circularity = (4 * np.pi * a / (peri * peri)) if peri else 0
        if n in (3, 4) or (n >= 7 and circularity >= 0.7):
            cv2.drawContours(output, [cnt], -1, 255, thickness=cv2.FILLED)
    return output


class MotionTracker:
    """Déplacement intermédiaire des rectangles via flux optique, non garanti."""

    def __init__(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.boxes: list[Box] = []
        self.last_frame: int | None = None

    @staticmethod
    def _small_gray(frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        ratio = min(1.0, 960 / w)
        if ratio < 1:
            frame = cv2.resize(frame, (round(w * ratio), round(h * ratio)))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), ratio

    def update(self, frame: np.ndarray, frame_number: int,
               redetect: bool, detector) -> list[Box]:
        gray, ratio = self._small_gray(frame)
        if redetect or self.last_frame is None or self.last_frame + 1 != frame_number:
            self.boxes = detector(frame)
        elif self.previous_gray is not None and self.boxes:
            updated: list[Box] = []
            for (x, y, w, h) in self.boxes:
                xx, yy, ww, hh = [round(v * ratio) for v in (x, y, w, h)]
                xx, yy = max(0, xx), max(0, yy)
                x2 = min(self.previous_gray.shape[1], xx + max(1, ww))
                y2 = min(self.previous_gray.shape[0], yy + max(1, hh))
                if x2 - xx < 6 or y2 - yy < 6:
                    updated.append((x, y, w, h))
                    continue
                roi = self.previous_gray[yy:y2, xx:x2]
                corners = cv2.goodFeaturesToTrack(roi, maxCorners=25,
                                                   qualityLevel=0.02, minDistance=4)
                if corners is None:
                    updated.append((x, y, w, h))
                    continue
                p0 = corners.astype(np.float32) + np.array([xx, yy], np.float32)
                p1, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray,
                                                          p0, None, winSize=(17, 17))
                if p1 is None or status is None:
                    updated.append((x, y, w, h))
                    continue
                good = status.reshape(-1) == 1
                if good.sum() < 3:
                    updated.append((x, y, w, h))
                    continue
                shift = np.median((p1[good] - p0[good]).reshape(-1, 2), axis=0) / ratio
                dx = float(np.clip(shift[0], -w / 2, w / 2))
                dy = float(np.clip(shift[1], -h / 2, h / 2))
                clipped = _clip_box((round(x + dx), round(y + dy), w, h),
                                    frame.shape[1], frame.shape[0])
                if clipped:
                    updated.append(clipped)
            self.boxes = updated
        self.previous_gray = gray
        self.last_frame = frame_number
        return list(self.boxes)


class VideoProcessor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.face_detector = HighRecallFaceDetector(require_scrfd=(settings.face_quality != "standard")) if (settings.faces and not settings.only_selected_subjects) else None
        self.yolo: YoloDetector | None = None
        self.yolo_error = ""
        if settings.yolo_classes:
            try:
                self.yolo = YoloDetector()
            except Exception as exc:
                # Avertissement explicite (jamais de faux « succès » pour l'IA).
                self.yolo_error = str(exc)
        self.head_detector = None
        self.head_error = ''
        if settings.heads and not settings.only_selected_subjects:
            try:
                if self.yolo is None and MODEL_PATH.exists():
                    self.yolo = YoloDetector()
                self.head_detector = HeadDetector(self.yolo)
            except Exception as exc:
                self.head_error = str(exc)
        self.person_detector = None
        self.person_error = ''
        if settings.persons and not settings.only_selected_subjects:
            try:
                if self.yolo is None and MODEL_PATH.exists():
                    self.yolo = YoloDetector()
                if (settings.body_silhouette or settings.body_shape != "box") and not PERSON_SEG_PATH.is_file():
                    raise FileNotFoundError('Silhouette ONNX absente : lancer INSTALLER.bat.')
                self.person_detector = PersonDetector(self.yolo, segmentation=(settings.body_silhouette or settings.body_shape != "box"))
            except Exception as exc:
                self.person_error = str(exc)
        self.person_tracker = MotionTracker()
        self.head_tracker = MotionTracker()
        self.face_tracker = MotionTracker()
        self.object_tracker = MotionTracker()

    def process(self, source: np.ndarray, frame_number: int = 0,
                preview: bool = False, face_boxes: list[Box] | None = None,
                body_boxes: list[Box] | None = None,
                custom_boxes: list[tuple[Box, bool]] | None = None) -> np.ndarray:
        """Applique les masques sur une COPIE, dimensions/pixels hors masque intacts."""
        cfg = self.settings
        output = source.copy()
        stride = max(1, cfg.inference_stride)
        refresh = preview or (frame_number % stride == 0)
        if (cfg.faces or cfg.heads) and (self.face_detector or face_boxes is not None):
            if face_boxes is not None:
                faces = face_boxes
            elif preview:
                faces = self.face_detector.detect(source, cfg.face_quality) if self.face_detector else []
            elif self.face_detector:
                faces = self.face_tracker.update(
                    source, frame_number, refresh, lambda f: self.face_detector.detect(f, cfg.face_quality))
            else:
                faces = []
            if face_boxes is not None and cfg.fail_closed and not faces:
                output[:] = 0  # échec de détection => image entière masquée
            for box in faces:
                _apply_head_box(output, box, cfg.mode, cfg.head_padding, cfg.head_shape, cfg.block_size)
        if cfg.heads and self.head_detector and face_boxes is None:
            finder = lambda f: self.head_detector.detect(f)
            if preview:
                heads = finder(source)
            else:
                heads = self.head_tracker.update(source, frame_number, refresh, finder)
            if cfg.fail_closed and not heads and not cfg.faces:
                output[:] = 0
            for box in heads:
                _apply_head_box(output, box, cfg.mode, cfg.head_padding * .42, cfg.head_shape, cfg.block_size)
        if cfg.persons and self.person_detector:
            if body_boxes is not None:
                bodies = body_boxes
            elif preview:
                bodies = self.person_detector.detect(source)
            else:
                bodies = self.person_tracker.update(
                    source, frame_number, refresh,
                    lambda frame: self.person_detector.detect(frame))
            if cfg.fail_closed and body_boxes is not None and not bodies and not cfg.faces and not cfg.heads:
                output[:] = 0
            if cfg.body_silhouette or cfg.body_shape != "box":
                # Chaque piste temporelle a une boîte de repli : même si la
                # segmentation manque une silhouette sur cette image, la zone
                # n'est pas laissée visible.
                segments = self.person_detector.masks(source)
                used: set[int] = set()
                for body in bodies:
                    body_area = max(1, body[2] * body[3])
                    prefer_precise = cfg.body_shape in ('blocks', 'silhouette') or (cfg.body_shape == 'auto' and body_area < (source.shape[0] * source.shape[1]) * 0.045)
                    choices = [(idx, iou(body, seg.box)) for idx, seg in enumerate(segments)
                               if idx not in used and iou(body, seg.box) >= .24]
                    if choices and prefer_precise:
                        idx, _ = max(choices, key=lambda pair: pair[1])
                        used.add(idx)
                        mask = segments[idx].mask
                        if cfg.body_padding:
                            # dilatation limitée et relative à la taille de la personne
                            radius = max(1, min(60, round(max(body[2],body[3]) * cfg.body_padding)))
                            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
                            mask = cv2.dilate(mask,k)
                        if np.any(mask):
                            if cfg.body_shape in ('blocks', 'auto'):
                                mask = tiled_silhouette(mask, cfg.block_size,
                                     dilation=max(1, min(18, round(max(body[2],body[3]) * cfg.body_padding * .20))))
                            _apply_binary_mask(output, mask, cfg.mode)
                        else:
                            _apply_box(output, body, cfg.mode, pad=cfg.body_padding)
                    else:
                        _apply_box(output, body, cfg.mode, pad=cfg.body_padding)
            else:
                for body in bodies:
                    _apply_box(output, body, cfg.mode, pad=cfg.body_padding)
        if cfg.yolo_classes and self.yolo:
            detector = lambda frame: self.yolo.detect(frame, cfg.yolo_classes, cfg.confidence)
            if preview:
                objects = detector(source)
            else:
                objects = self.object_tracker.update(source, frame_number, refresh, detector)
            for box in objects:
                _apply_box(output, box, cfg.mode, pad=0.08)
        # Objet choisi librement : le plan d'export contient les positions suivies
        # sur toutes les images. L'aperçu n'affiche que la sélection sur son image.
        selected_custom = (custom_boxes if custom_boxes is not None
             else [(obj.box, obj.contour) for obj in cfg.custom_objects if obj.frame == frame_number])
        for box, contour in selected_custom:
            if contour and cfg.body_shape != 'box':
                mask = grabcut_selected(source, box, margin=max(2, round(max(box[2],box[3])*.025)))
                if np.any(mask):
                    if cfg.body_shape in ('blocks', 'auto'):
                        mask = tiled_silhouette(mask, cfg.block_size,
                                                dilation=max(1, cfg.block_size // 6))
                    _apply_binary_mask(output, mask, cfg.mode)
                else:
                    _apply_box(output,box,cfg.mode,.04)
            else:
                _apply_box(output,box,cfg.mode,.04)
        for region in cfg.regions:
            _apply_box(output, region, cfg.mode)
        if cfg.color:
            _apply_binary_mask(output, color_mask(source, cfg.color_bgr,
                                                  cfg.hue_tolerance), cfg.mode)
        if cfg.shapes:
            _apply_binary_mask(output, shape_mask(source), cfg.mode)
        return output
