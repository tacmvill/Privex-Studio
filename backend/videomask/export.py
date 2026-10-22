"""Export local via FFmpeg embarqué (imageio-ffmpeg).

Pour les sources à cadence CONSTANTE (CFR), produit le même nombre d'images,
la même résolution et la même cadence. Une source VFR doit être prise en charge
par un pipeline timestamps/PTS dédié (hors périmètre V1).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from typing import Callable
import os
import subprocess

import cv2
import imageio_ffmpeg

from .core import Settings, VideoProcessor
from .temporal import scan_subjects
from .custom_tracking import plan_custom_objects


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


def inspect_video(path: str) -> VideoInfo:
    video = cv2.VideoCapture(path)
    try:
        if not video.isOpened():
            raise ValueError("Impossible d'ouvrir la vidéo. Codec ou fichier non pris en charge.")
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(video.get(cv2.CAP_PROP_FPS))
        frames = max(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width < 1 or height < 1 or not 0.1 <= fps <= 480:
            raise ValueError("Métadonnées vidéo invalides (dimensions ou FPS).")
        return VideoInfo(width, height, fps, frames)
    finally:
        video.release()


def _ffmpeg_fps(fps: float) -> str:
    # Ramène 29.9700299 à 30000/1001 sans modifier la cadence effective.
    return str(Fraction(fps).limit_denominator(100100))


@lru_cache(maxsize=2)
def _amd_amf_available(exe: str) -> bool:
    """Essai réel d'encodage, pas seulement présence de l'encodeur dans la liste.

    Certains pilotes/APU déclarent h264_amf mais ne peuvent pas l'initialiser.
    """
    creationflags = getattr(subprocess, 'CREATE_NO_WINDOW',0)
    cmd = [exe,'-hide_banner','-loglevel','error','-y','-f','rawvideo',
           '-pixel_format','bgr24','-video_size','64x64','-framerate','1',
           '-i','pipe:0','-frames:v','1','-c:v','h264_amf','-pix_fmt',
           'yuv420p','-f','null','-']
    try:
        result = subprocess.run(cmd,input=bytes(64*64*3),stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,timeout=12,
                                creationflags=creationflags)
        return result.returncode == 0
    except (OSError,subprocess.TimeoutExpired):
        return False


def export_video(input_path: str, output_path: str, settings: Settings,
                 quality: str = "high", progress: Callable[[int], None] | None = None,
                 cancel: Callable[[], bool] | None = None,
                 start_frame: int = 0, max_frames: int | None = None,
                 include_audio: bool = True, audio_volume: float = 1.0) -> tuple[int, str]:
    """Retourne (nombre_images, fournisseur_IA). Émet des erreurs claires.

    quality : 'high' (MP4 H.264) ou 'lossless' (FFV1 MKV).
    """
    if quality not in {"high", "fast", "amd", "lossless"}:
        raise ValueError("Qualité d'export invalide.")
    if os.path.normcase(os.path.abspath(input_path)) == os.path.normcase(os.path.abspath(output_path)):
        raise ValueError("La sortie ne peut pas écraser la vidéo source.")
    if not 0 <= audio_volume <= 3:
        raise ValueError("Volume audio hors limites (0 à 300 %).")
    include_audio = include_audio and audio_volume > 0
    info = inspect_video(input_path)
    if start_frame < 0 or (max_frames is not None and max_frames < 1):
        raise ValueError('Séquence vidéo invalide.')
    if start_frame and info.frames and start_frame >= info.frames:
        raise ValueError('Début de séquence au-delà de la vidéo.')
    selected_frames = (max(0,min(info.frames-start_frame, max_frames))
                       if max_frames is not None and info.frames else max_frames or info.frames)
    reader = cv2.VideoCapture(input_path)
    if not reader.isOpened():
        raise RuntimeError("Impossible de relire la vidéo.")
    if start_frame:
        reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    process = None
    target = Path(output_path)
    count = 0
    complete = False
    try:
        processor = VideoProcessor(settings)
        if settings.yolo_classes and not processor.yolo:
            raise RuntimeError("Détection IA sélectionnée mais indisponible : " + processor.yolo_error)

        if settings.heads and not settings.only_selected_subjects and processor.head_detector is None:
            raise RuntimeError('Tête/dos non disponible : ' + processor.head_error)
        face_plan = None
        person_plan = None
        if settings.persons and not settings.only_selected_subjects and processor.person_detector is None:
            raise RuntimeError('Personnes non disponibles : ' + processor.person_error)
        if not settings.only_selected_subjects and (settings.faces or settings.heads or settings.persons):
            # Analyse sans compromis : un passage intégral sur toutes les images
            # puis interpolation temporelle AVANT le second passage d'encodage.
            face_plan, person_plan = scan_subjects(
                reader, processor.face_detector, settings.face_quality,
                progress=progress, cancel=cancel, estimated=selected_frames,
                head_detector=processor.head_detector,
                person_detector=processor.person_detector,
                max_frames=max_frames, fps=info.fps)
            if not face_plan:
                raise RuntimeError('Aucune image décodable pendant la détection.')
            reader.release()
            reader = cv2.VideoCapture(input_path)
            if not reader.isOpened():
                raise RuntimeError('Impossible de rouvrir la vidéo pour exporter.')
            if start_frame:
                reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # Les objets personnalisés sont suivis de leur image-ancre vers le début
        # ET vers la fin. Ce plan est calculé indépendamment de la passe IA.
        custom_plan = None
        if settings.custom_objects:
            if start_frame or max_frames is not None:
                raise ValueError('Le suivi personnalisé exige la vidéo complète.')
            if not info.frames:
                raise ValueError('Nombre d’images requis pour le suivi personnalisé.')
            auto_scan = face_plan is not None
            def custom_progress(value:int):
                if progress:
                    start, span = (60, 10) if auto_scan else (0, 55)
                    progress(min(99, start + round(span * value / 100)))
            custom_plan = plan_custom_objects(input_path, settings.custom_objects, info.frames,
                                             progress=custom_progress, cancel=cancel)
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        use_amf = quality == 'amd' and _amd_amf_available(exe)
        if use_amf:
            codec_args = ['-c:v','h264_amf','-quality','speed','-rc','cqp',
                          '-qp_i','17','-qp_p','17','-pix_fmt','yuv420p']
        elif quality in ('high','fast','amd'):
            # Fast accélère l'ENCODAGE uniquement, sans sauter de détections ni
            # d'images. Il peut augmenter la taille du fichier à CRF identique.
            codec_args = ['-c:v','libx264','-preset',
                          'veryfast' if quality in ('fast','amd') else 'medium',
                          '-crf','15','-pix_fmt','yuv420p']
        else:
            codec_args = ["-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgr0"]

        args = [exe, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
                f"{info.width}x{info.height}", "-framerate", _ffmpeg_fps(info.fps),
                '-i','pipe:0']
        if include_audio:
            if start_frame:
                args += ['-ss',str(start_frame/info.fps)]
            args += ['-i',input_path]
            if max_frames is not None:
                args += ['-t',str(max_frames/info.fps)]
        args += ['-map','0:v:0']
        if include_audio:
            args += ['-map','1:a?','-map_metadata','1']
        args += [*codec_args]
        if include_audio:
            if abs(audio_volume - 1.) < .001:
                args += ['-c:a','copy']
            else:
                args += ['-af',f'volume={audio_volume:.3f}', '-c:a','aac','-b:a','192k']
        args += ['-sn','-dn',str(target)]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, creationflags=creationflags)
        while True:
            if max_frames is not None and count >= max_frames:
                break
            if cancel and cancel():
                raise InterruptedError("Export annulé.")
            ret, frame = reader.read()
            if not ret:
                break
            if face_plan is not None and count >= len(face_plan):
                raise RuntimeError('Le décodage diffère entre les deux passes ; export arrêté.')
            rendered = processor.process(
                frame, count,
                face_boxes=face_plan[count] if face_plan is not None else None,
                body_boxes=person_plan[count] if person_plan is not None else None,
                custom_boxes=custom_plan[count] if custom_plan is not None else None)
            assert process.stdin is not None
            try:
                process.stdin.write(rendered.tobytes())
            except BrokenPipeError as exc:
                detail = (process.stderr.read().decode(errors="replace") if process.stderr else "")
                raise RuntimeError("FFmpeg a interrompu l'export : " + detail[-1000:]) from exc
            count += 1
            if progress and selected_frames:
                base = 70 if face_plan is not None and custom_plan is not None else (
                    60 if face_plan is not None else (55 if custom_plan is not None else 0))
                progress(min(99, base + round((99-base) * count / selected_frames)))

        if face_plan is not None and count != len(face_plan):
            raise RuntimeError('Nombre d’images incohérent entre analyse et rendu, fichier rejeté.')
        if person_plan is not None and count != len(person_plan):
            raise RuntimeError('Nombre d’images de corps incohérent ; fichier rejeté.')
        if process.stdin:
            process.stdin.close()
        ret = process.wait()
        error = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if ret != 0:
            raise RuntimeError("FFmpeg n'a pas pu créer la sortie : " + error[-1600:])
        if count == 0:
            raise RuntimeError("Aucune image décodée.")
        complete = True
        if progress:
            progress(100)
        return count, (
            'Visages : ' + (processor.face_detector.provider if processor.face_detector else 'désactivés') +
            (' | Têtes : ' + processor.head_detector.provider if processor.head_detector else '') +
            (' | Corps : ' + processor.person_detector.provider if processor.person_detector else '') +
            (' | Encodeur : AMD AMF' if use_amf else ' | Encodeur : x264' + (' (repli AMF)' if quality == 'amd' else '')) +
            (' | Objets : ' + processor.yolo.provider if processor.yolo and settings.yolo_classes else '')
        )
    finally:
        reader.release()
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.stderr:
                process.stderr.close()
        if not complete and target.exists():
            try:
                target.unlink()
            except OSError:
                pass


def export_image(input_path: str, output_path: str, settings: Settings) -> str:
    """Export d'une image unique, sans rééchantillonnage géométrique."""
    import numpy as np
    source=cv2.imdecode(np.fromfile(input_path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError('Image illisible.')
    ext=Path(output_path).suffix.lower()
    if ext not in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}:
        raise ValueError('Format image de sortie non pris en charge.')
    out=VideoProcessor(settings).process(source,frame_number=0,preview=True)
    args=[cv2.IMWRITE_JPEG_QUALITY,98] if ext in {'.jpg','.jpeg'} else []
    ok,data=cv2.imencode(ext,out,args)
    if not ok: raise RuntimeError('Encodage de l’image impossible.')
    Path(output_path).write_bytes(data.tobytes())
    return output_path
