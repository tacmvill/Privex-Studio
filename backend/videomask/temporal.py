"""Plan temporel en deux passes : comble des trous courts sans images vidéo en RAM.

Aucun suivi ne peut reconstruire les visages jamais détectés. Les changements
brusques de plan coupent toutes les extrapolations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Callable
import cv2
import numpy as np

Box = tuple[int, int, int, int]


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0],b[0]), max(a[1],b[1])
    x2, y2 = min(a[0]+a[2],b[0]+b[2]), min(a[1]+a[3],b[1]+b[3])
    intersection = max(0,x2-x1)*max(0,y2-y1)
    union = a[2]*a[3]+b[2]*b[3]-intersection
    return intersection/union if union else 0


def _match(a: Box, b: Box) -> bool:
    if _iou(a,b) >= .10:
        return True
    ax,ay = a[0]+a[2]/2,a[1]+a[3]/2
    bx,by = b[0]+b[2]/2,b[1]+b[3]/2
    distance = np.hypot(ax-bx,ay-by)
    size = max(a[2],a[3],b[2],b[3])
    ratio = max(a[2]/max(1,b[2]),b[2]/max(1,a[2]))
    return distance <= size*.65 and ratio <= 2.0


def _lerp(a: Box, b: Box, ratio: float) -> Box:
    return tuple(round(x*(1-ratio)+y*ratio) for x,y in zip(a,b))  # type: ignore[return-value]


@dataclass
class _Track:
    hits: list[tuple[int,Box]] = field(default_factory=list)
    @property
    def latest(self):
        return self.hits[-1]


def fill_short_gaps(detected: list[list[Box]], cuts: set[int] | None = None,
                    max_gap: int = 12, hold: int = 3) -> list[list[Box]]:
    """Détections image-à-image -> boîtes avec interpolation bidirectionnelle.

    `cuts` contient l'index de la première image APRES chaque coupe.
    Ne traverse jamais ces frontières. Chaque sortie reprend d'abord les
    détections brutes pour ne pas effacer un visage réel.
    """
    cuts = cuts or set()
    result = [list(boxes) for boxes in detected]
    ends = sorted(c for c in cuts if 0 < c < len(detected)) + [len(detected)]
    start = 0
    for end in ends:
        tracks: list[_Track] = []
        for index in range(start,end):
            used: set[int] = set()
            for box in detected[index]:
                candidates = [(ti, _iou(track.latest[1], box)) for ti,track in enumerate(tracks)
                    if ti not in used and index-track.latest[0] <= max_gap and _match(track.latest[1],box)]
                if candidates:
                    # Association la plus proche (IoU favorisé)
                    ti = max(candidates,key=lambda q:q[1])[0]
                    tracks[ti].hits.append((index,box))
                    used.add(ti)
                else:
                    tracks.append(_Track([(index,box)]))
                    used.add(len(tracks)-1)
        for track in tracks:
            hits = track.hits
            for (a,ba),(b,bb) in zip(hits,hits[1:]):
                if 1 < b-a <= max_gap:
                    for t in range(a+1,b):
                        result[t].append(_lerp(ba,bb,(t-a)/(b-a)))
            # Fait déborder quelques images AVANT l'apparition et APRÈS la disparition,
            # mais pas au-delà d'une coupe détectée. Privilégie un sur-masquage.
            a,ba = hits[0]
            b,bb = hits[-1]
            for t in range(max(start,a-hold),a):
                result[t].append(ba)
            for t in range(b+1,min(end,b+hold+1)):
                result[t].append(bb)
        start = end
    return result


def scene_cut(prev: np.ndarray | None, frame: np.ndarray) -> tuple[bool,np.ndarray]:
    small = cv2.resize(frame, (128,72), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if prev is None:
        return False, gray
    # seuil prudent : un mouvement violent peut aussi couper une trajectoire.
    changed = float(np.mean(cv2.absdiff(gray,prev)))
    return changed >= 65., gray


def _optical_prediction(previous: np.ndarray, current: np.ndarray,
                        box: Box) -> Box:
    """Suggère une translation depuis les points KLT à l'intérieur du masque.

    Lorsque les points sont absents (flou intense), retient temporairement la
    dernière position. Ces prédictions sont limitées à 3 images successives.
    """
    h, w = previous.shape
    x, y, bw, bh = box
    x1, y1 = max(0,x), max(0,y)
    x2, y2 = min(w,x+bw), min(h,y+bh)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return box
    mask = np.zeros_like(previous, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    corners = cv2.goodFeaturesToTrack(previous, maxCorners=36, qualityLevel=.015,
                                      minDistance=3, mask=mask)
    if corners is None or len(corners) < 3:
        return box
    nxt, statuses, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, corners, None, winSize=(21,21), maxLevel=3)
    if nxt is None or statuses is None:
        return box
    valid = statuses.reshape(-1) == 1
    if valid.sum() < 3:
        return box
    movements = (nxt[valid] - corners[valid]).reshape(-1,2)
    dx, dy = np.median(movements, axis=0)
    if not np.isfinite([dx,dy]).all():
        return box
    # Un mouvement plus large que le visage est trop incertain.
    if abs(dx) > max(8,bw*.8) or abs(dy) > max(8,bh*.8):
        return box
    nx = max(0, min(w-bw, round(x + float(dx))))
    ny = max(0, min(h-bh, round(y + float(dy))))
    return nx,ny,bw,bh


def _near(a: Box, b: Box) -> bool:
    return _iou(a,b) > .15 or _match(a,b)


class _MaskStream:
    """Suivi par classe de sujet (têtes OU personnes), jamais mélangé.

    Les boîtes sont uniquement une trajectoire de masquage conservatrice ;
    aucune identité biométrique n'est inférée.
    """
    def __init__(self, fps: float):
        self.max_hold = max(4, min(25, round(fps * .65)))
        self.detected: list[list[Box]] = []
        self.cuts: set[int] = set()
        self.buffer: deque[tuple[int, np.ndarray, float]] = deque(maxlen=self.max_hold + 1)
        self.tracks: list[tuple[Box, int]] = []
        self.previous_gray: np.ndarray | None = None

    def push(self, raw: list[Box], gray: np.ndarray, scale: float, cut: bool) -> None:
        index = len(self.detected)
        if cut:
            self.cuts.add(index)
            self.tracks.clear()
            self.buffer.clear()
        combined = list(raw)
        fresh_tracks: list[tuple[Box, int]] = [(box, 0) for box in raw]
        if self.previous_gray is not None and not cut:
            for old, age in self.tracks:
                if age >= self.max_hold:
                    continue
                scaled_old = tuple(round(v * scale) for v in old)
                prediction = _optical_prediction(self.previous_gray, gray, scaled_old)
                predicted = tuple(round(v / scale) for v in prediction)
                if not any(_near(predicted, b) for b in raw):
                    combined.append(predicted)
                    if not any(_near(predicted, b) for b, _ in fresh_tracks):
                        fresh_tracks.append((predicted, age + 1))
        # Remonter depuis une détection tardive vers les images antérieures.
        if raw and self.buffer:
            for box in raw:
                if self.tracks and any(_near(box, prior) for prior, _ in self.tracks):
                    continue
                current_box = tuple(round(v * scale) for v in box)
                current_gray = gray
                for past_index, past_gray, past_scale in reversed(self.buffer):
                    if past_index in self.cuts or past_scale != scale:
                        break
                    old_box = _optical_prediction(current_gray, past_gray, current_box)
                    back_box = tuple(round(v / scale) for v in old_box)
                    if not any(_near(back_box, existing) for existing in self.detected[past_index]):
                        self.detected[past_index].append(back_box)
                    current_gray, current_box = past_gray, old_box
        self.detected.append(combined)
        self.tracks = fresh_tracks
        self.previous_gray = gray
        self.buffer.append((index, gray, scale))

    def finish(self) -> list[list[Box]]:
        return fill_short_gaps(self.detected, self.cuts,
                               max_gap=self.max_hold, hold=min(self.max_hold, 8))


def scan_subjects(reader, detector=None, quality: str = 'extreme',
                  progress: Callable[[int], None] | None = None,
                  cancel: Callable[[], bool] | None = None,
                  estimated: int = 0, head_detector=None, person_detector=None,
                  max_frames: int | None = None, fps: float = 30.0
                  ) -> tuple[list[list[Box]], list[list[Box]]]:
    """Détection unique par image, suivi indépendant des têtes et des corps.

    Mémoire proportionnelle au nombre de boîtes, et non aux pixels de la vidéo.
    Renvoie deux plans dans l'ordre (visages/têtes, personnes). Les résultats
    ne doivent PAS être considérés comme une anonymisation garantie.
    """
    if detector is None and head_detector is None and person_detector is None:
        raise ValueError('Un détecteur de tête, visage ou personne est nécessaire.')
    heads = _MaskStream(fps)
    people = _MaskStream(fps)
    prev_scene = None
    count = 0
    while max_frames is None or count < max_frames:
        if cancel and cancel():
            raise InterruptedError('Analyse annulée.')
        ok, frame = reader.read()
        if not ok:
            break
        cut, prev_scene = scene_cut(prev_scene, frame)
        h, w = frame.shape[:2]
        scale = min(1., 640 / w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        found_heads = list(detector.detect(frame, quality)) if detector is not None else []
        if head_detector is not None:
            found_heads.extend(head_detector.detect(frame))
        found_people = list(person_detector.detect(frame)) if person_detector is not None else []
        heads.push(found_heads, gray, scale, cut)
        people.push(found_people, gray, scale, cut)
        count += 1
        if progress and estimated and count % 5 == 0:
            progress(min(60, round(60 * count / estimated)))
    return heads.finish(), people.finish()


def scan_faces(reader, detector, quality: str = 'extreme',
               progress: Callable[[int], None] | None = None,
               cancel: Callable[[], bool] | None = None,
               estimated: int = 0, head_detector=None,
               max_frames: int | None = None, fps: float = 30.0) -> list[list[Box]]:
    """API V4 conservée pour les tests et scripts existants."""
    return scan_subjects(reader, detector, quality, progress, cancel, estimated,
                         head_detector=head_detector, max_frames=max_frames,
                         fps=fps)[0]
