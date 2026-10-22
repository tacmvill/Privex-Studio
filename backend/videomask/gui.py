"""VideoMask AI V9 Contour Studio — test sur trois images, sans lecteur vidéo.

100 % traitement local. L'installation initiale des modèles nécessite Internet.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QCheckBox, QComboBox,
    QSlider, QSpinBox, QProgressBar, QScrollArea, QFrame, QGraphicsDropShadowEffect,
    QDialog,
)

from .core import Settings, VideoProcessor, MODEL_PATH
from .faces import SCRFD_PATH, YUNET_PATH
from .heads import HEAD_MODEL_PATH
from .persons import PERSON_SEG_PATH
from .custom_tracking import CustomObject
from .export import inspect_video, export_video, export_image, VideoInfo

IMAGE_EXT={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}


def style_sheet(dark: bool):
    """Design original PySide6 inspiré des bibliothèques UI modernes.

    Palette/densité adaptées à Windows; aucun composant React tiers importé.
    """
    c = ({"bg":"#080D16", "card":"#111927", "card2":"#182232", "input":"#0C1422",
          "edge":"#2A3647", "text":"#F4F7FC", "muted":"#9AAAC0", "accent":"#75D6C8",
          "hover":"#233447", "green":"#79E5CB", "soft":"#B4B9FF", "buttonText":"#071A21"}
         if dark else
         {"bg":"#EAF0F7", "card":"#FFFFFF", "card2":"#F2F5FA", "input":"#F8FAFE",
          "edge":"#D7E0ED", "text":"#16243A", "muted":"#607189", "accent":"#147F87",
          "hover":"#DFEAF2", "green":"#007E6F", "soft":"#5759BE", "buttonText":"#FFFFFF"})
    q=lambda name:c[name]
    sheet=f"""
QMainWindow, QWidget#surface {{background:{q('bg')};color:{q('text')};}}
QWidget {{font-family:'Segoe UI';font-size:12px;color:{q('text')};}}
QFrame#card {{background:{q('card')};border:1px solid {q('edge')};border-radius:22px;}}
QFrame#subcard {{background:{q('card2')};border:1px solid {q('edge')};border-radius:16px;}}
QLabel#heading {{font-size:27px;font-weight:800;letter-spacing:.4px;color:{q('text')};}}
QLabel#section {{font-size:15px;font-weight:750;color:{q('text')};}}
QLabel#muted {{color:{q('muted')};font-size:12px;}}
QLabel#metric {{color:{q('accent')};font-size:12px;font-weight:750;}}
QLabel#eyebrow {{color:{q('soft')};font-size:11px;font-weight:800;}}
QPushButton {{background:{q('card2')};color:{q('text')};border:1px solid {q('edge')};border-radius:12px;padding:11px 14px;font-weight:650;min-height:19px;}}
QPushButton:hover {{background:{q('hover')};border:1px solid {q('accent')};}}
QPushButton:disabled {{color:{q('muted')};background:{q('input')};}}
QPushButton#primary {{background:{q('accent')};color:{q('buttonText')};border:0;font-weight:800;}}
QPushButton#primary:hover {{background:{q('green')};}}
QPushButton#outlined {{border:1px solid {q('accent')};color:{q('accent')};font-weight:750;}}
QPushButton#selected {{background:{q('soft')};color:{q('bg')};font-weight:800;}}
QComboBox,QSpinBox {{background:{q('input')};border:1px solid {q('edge')};padding:10px 11px;border-radius:12px;min-height:19px;}}
QComboBox QAbstractItemView {{background:{q('card2')};color:{q('text')};selection-background-color:{q('accent')};}}
QCheckBox {{spacing:10px;}}
QCheckBox::indicator {{width:17px;height:17px;}}
QCheckBox::indicator:unchecked {{background:{q('input')};border:1px solid {q('edge')};border-radius:5px;}}
QCheckBox::indicator:checked {{background:{q('accent')};border:1px solid {q('accent')};border-radius:5px;}}
QSlider::groove:horizontal {{height:6px;background:{q('edge')};border-radius:3px;}}
QSlider::handle:horizontal {{background:{q('accent')};width:17px;margin:-6px 0;border-radius:9px;}}
QProgressBar {{background:{q('input')};border:1px solid {q('edge')};border-radius:8px;text-align:center;height:18px;}}
QProgressBar::chunk {{background:{q('accent')};border-radius:7px;}}
QScrollArea {{border:0;background:transparent;}}
QScrollBar:vertical {{background:transparent;width:9px;}}
QScrollBar::handle:vertical {{background:{q('edge')};border-radius:4px;}}
"""
    return sheet,c


def card(title: str|None=None, sub: str|None=None):
    f=QFrame();f.setObjectName('card')
    layout=QVBoxLayout(f);layout.setContentsMargins(20,19,20,19);layout.setSpacing(14)
    if title:
        label=QLabel(title);label.setObjectName('section');layout.addWidget(label)
    if sub:
        desc=QLabel(sub);desc.setObjectName('muted');desc.setWordWrap(True);layout.addWidget(desc)
    return f,layout


def shadow(widget):
    # Ombre externe discrète ; la bordure interne simule le relief neumorphique.
    effect=QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(27);effect.setOffset(4,7);effect.setColor(QColor(0,0,0,43))
    widget.setGraphicsEffect(effect)


def label_text(t: str, kind='muted'):
    label=QLabel(t);label.setObjectName(kind);label.setWordWrap(True)
    return label


class PreviewCanvas(QLabel):
    regionSelected=Signal(int,int,int,int)
    sampled=Signal(int,int)
    clicked=Signal()
    def __init__(self):
        super().__init__('Aucun média')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(215,160)
        self.setStyleSheet('background:#090F1C;border-radius:10px;')
        self.frame=None;self.bounds=QRect();self.origin=None;self.endpoint=None
        self.tool='view'
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def show_bgr(self,img):
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w=rgb.shape[:2]
        self.frame=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format.Format_RGB888).copy()
        self.redraw()

    def redraw(self):
        if self.frame is None:return
        fitted=self.frame.size().scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio)
        self.bounds=QRect((self.width()-fitted.width())//2,(self.height()-fitted.height())//2,fitted.width(),fitted.height())
        self.setPixmap(QPixmap.fromImage(self.frame).scaled(fitted,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self,event):
        super().resizeEvent(event);self.redraw()

    def map_xy(self,pos):
        if self.frame is None or not self.bounds.contains(pos):return None
        return QPoint(max(0,min(self.frame.width()-1,round((pos.x()-self.bounds.x())*self.frame.width()/max(1,self.bounds.width())))),
                      max(0,min(self.frame.height()-1,round((pos.y()-self.bounds.y())*self.frame.height()/max(1,self.bounds.height())))))

    def mousePressEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton:return
        p=self.map_xy(event.position().toPoint())
        if p is None:return
        if self.tool=='color': self.sampled.emit(p.x(),p.y())
        elif self.tool=='view': self.clicked.emit()
        else:self.origin=p;self.endpoint=p

    def mouseMoveEvent(self,event):
        if self.origin is None:return
        p=self.map_xy(event.position().toPoint())
        if p is not None:self.endpoint=p;self.update()

    def mouseReleaseEvent(self,event):
        if self.origin is None:return
        a=self.origin;b=self.map_xy(event.position().toPoint()) or self.endpoint
        self.origin=None;self.endpoint=None;self.update()
        if b is None:return
        x1,x2=sorted((a.x(),b.x()));y1,y2=sorted((a.y(),b.y()))
        if x2-x1>=6 and y2-y1>=6:self.regionSelected.emit(x1,y1,x2-x1,y2-y1)

    def paintEvent(self,event):
        super().paintEvent(event)
        if self.origin is None or self.endpoint is None or self.frame is None:return
        sx=self.bounds.width()/self.frame.width();sy=self.bounds.height()/self.frame.height()
        a=self.origin;b=self.endpoint
        r=QRect(QPoint(round(self.bounds.x()+a.x()*sx),round(self.bounds.y()+a.y()*sy)),
                QPoint(round(self.bounds.x()+b.x()*sx),round(self.bounds.y()+b.y()*sy))).normalized()
        painter=QPainter(self)
        painter.setPen(QPen(QColor('#B6A7FE'),2,Qt.PenStyle.DashLine))
        painter.setBrush(QColor(155,141,251,60));painter.drawRect(r)



class ZoomDialog(QDialog):
    def __init__(self, title:str, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1120, 820)
        self.factor = 1.0
        self.base = frame.copy()
        outer = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.info = QLabel()
        toolbar.addWidget(self.info)
        toolbar.addStretch()
        for text, slot in (('−', self.zoom_out), ('100 %', self.zoom_reset), ('＋', self.zoom_in)):
            b = QPushButton(text); b.clicked.connect(slot); toolbar.addWidget(b)
        fs = QPushButton('Plein écran'); fs.clicked.connect(self.toggle_fullscreen); toolbar.addWidget(fs)
        outer.addLayout(toolbar)
        self.area = QScrollArea(); self.area.setWidgetResizable(False)
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.area.setWidget(self.label); outer.addWidget(self.area, stretch=1)
        self.render()

    def render(self):
        rgb = cv2.cvtColor(self.base, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(img)
        scaled = pix.scaled(max(1, round(pix.width()*self.factor)), max(1, round(pix.height()*self.factor)),
                            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)
        self.label.setFixedSize(scaled.size())
        self.info.setText(f'{w}×{h}  ·  zoom {round(self.factor*100)} %')

    def zoom_in(self):
        self.factor = min(8.0, self.factor * 1.25); self.render()
    def zoom_out(self):
        self.factor = max(0.1, self.factor / 1.25); self.render()
    def zoom_reset(self):
        self.factor = 1.0; self.render()
    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()


class PreviewThread(QThread):
    ready=Signal(int,object,object,str)
    failed=Signal(int,str)
    progressChanged=Signal(int)
    def __init__(self,generation:int,frames:list,indices:list[int],cfg:Settings,processor=None):
        super().__init__();self.generation=generation;self.frames=frames;self.indices=indices
        self.cfg=cfg;self.processor=processor
    def run(self):
        try:
            proc=self.processor or VideoProcessor(self.cfg)
            proc.settings=self.cfg
            result=[]
            for i,(frame,idx) in enumerate(zip(self.frames,self.indices)):
                result.append(proc.process(frame,int(idx),preview=True))
                self.progressChanged.emit(round((i+1)*100/max(1,len(self.frames))))
            self.ready.emit(self.generation,result,proc,'')
        except Exception as exc:
            self.failed.emit(self.generation,str(exc))


class ExportThread(QThread):
    progressChanged=Signal(int)
    completed=Signal(str,int,str)
    failed=Signal(str)
    canceled=Signal()
    def __init__(self,source,destination,cfg,quality,kind,volume):
        super().__init__();self.source=source;self.dest=destination;self.cfg=cfg
        self.quality=quality;self.kind=kind;self.volume=volume;self.abort=False
    def run(self):
        try:
            if self.kind=='image':
                export_image(self.source,self.dest,self.cfg)
                self.completed.emit(self.dest,1,'Image OpenCV')
            else:
                count,provider=export_video(self.source,self.dest,self.cfg,self.quality,
                    progress=self.progressChanged.emit,cancel=lambda:self.abort,
                    include_audio=self.volume>0,audio_volume=self.volume)
                self.completed.emit(self.dest,count,provider)
        except InterruptedError:self.canceled.emit()
        except Exception as exc:self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('VideoMask AI · Contour Studio V9')
        self.resize(1540,980);self.setMinimumSize(1090,740)
        self.dark=True;self.path='';self.kind=None;self.info:VideoInfo|None=None
        self.sample_frames=[];self.sample_indices=[];self.preview_outputs=[]
        self.settings=Settings(heads=True,persons=True,body_shape='blocks',head_shape='blocks',mode='blur')
        self.export_thread=None;self.preview_thread=None;self.preview_version=0
        self.cached_processor=None;self.cached_signature=None;self.active=False;self.show_original=False
        self.current_frame_index=0;self.navigator_frame=None;self.selecting_custom=False
        self.build_ui();self.refresh_theme();self.update_models()

    def refresh_theme(self):
        self.setStyleSheet(style_sheet(self.dark)[0])
        self.theme_btn.setText('☀  Mode clair' if self.dark else '☾  Mode sombre')

    def build_ui(self):
        base=QWidget();base.setObjectName('surface');self.setCentralWidget(base)
        outer=QVBoxLayout(base);outer.setContentsMargins(28,22,28,22);outer.setSpacing(20)
        header=QHBoxLayout()
        brand=QVBoxLayout();brand.setSpacing(2)
        h=QLabel('◈  VideoMask AI');h.setObjectName('heading');brand.addWidget(h)
        brand.addWidget(label_text('CONTOUR STUDIO    •    PRIVATE VIDEO WORKSPACE    •    NO CLOUD', 'eyebrow'))
        header.addLayout(brand);header.addStretch()
        header.addWidget(label_text('●  100 % local', 'metric'))
        self.theme_btn=QPushButton();self.theme_btn.clicked.connect(self.toggle_theme)
        header.addWidget(self.theme_btn)
        self.open_btn=QPushButton('＋  Importer image / vidéo')
        self.open_btn.setObjectName('primary');self.open_btn.clicked.connect(self.open_media)
        header.addWidget(self.open_btn);outer.addLayout(header)
        main=QHBoxLayout();main.setSpacing(21);outer.addLayout(main,stretch=1)
        left=QVBoxLayout();left.setSpacing(19);main.addLayout(left,stretch=1)
        hero,hl=card('Aperçu de votre média',
            'Trois images de contrôle. Cliquez sur une vignette pour l’ouvrir, zoomer et passer en plein écran.')
        shadow(hero)
        hero_scroll=QScrollArea(); hero_scroll.setWidgetResizable(True)
        hero_scroll.setWidget(hero)
        left.addWidget(hero_scroll,stretch=1)
        bar=QHBoxLayout();bar.addWidget(label_text('ÉCHANTILLONS · 3 PHOTOS DE LA VIDÉO','metric'));bar.addStretch()
        self.compare_btn=QPushButton('◫  Voir les originaux');self.compare_btn.clicked.connect(self.toggle_compare)
        bar.addWidget(self.compare_btn);hl.addLayout(bar)
        grid=QGridLayout();grid.setSpacing(16);hl.addLayout(grid,stretch=1)
        self.canvases=[];self.sample_badges=[]
        for i in range(3):
            frame,lay=card();frame.setObjectName('subcard')
            lay.setContentsMargins(9,9,9,9);lay.setSpacing(7)
            top=QHBoxLayout();top.addWidget(label_text(f'FRAME 0{i+1}','metric'));top.addStretch()
            tag=label_text('—');top.addWidget(tag);self.sample_badges.append(tag)
            lay.addLayout(top)
            canvas=PreviewCanvas(); canvas.tool='view'; lay.addWidget(canvas,stretch=1)
            canvas.clicked.connect(lambda j=i: self.open_zoom(j))
            canvas.setToolTip('Cliquez pour agrandir cette image')
            self.canvases.append(canvas)
            grid.addWidget(frame,0,i)
        actions=QHBoxLayout();actions.setSpacing(11)
        self.apply_btn=QPushButton('✓  Appliquer aux 3 images'); self.apply_btn.setObjectName('primary')
        self.apply_btn.clicked.connect(self.start_preview); self.apply_btn.setEnabled(False)
        actions.addWidget(self.apply_btn,stretch=1)
        self.compare_info=label_text('Cliquez sur une image pour l’examiner.', 'muted')
        actions.addWidget(self.compare_info)
        hl.addLayout(actions)
        nav,navl=card('Explorer la source','Naviguez jusqu’à n’importe quelle image puis choisissez « Ajouter un masque personnalisé » pour dessiner autour d’un sujet.')
        nav.setObjectName('subcard'); hl.addWidget(nav)
        self.nav_canvas=PreviewCanvas(); self.nav_canvas.setMinimumHeight(225); self.nav_canvas.tool="view"; navl.addWidget(self.nav_canvas, stretch=1)
        self.nav_canvas.regionSelected.connect(self.add_region_from_navigator)
        navbar=QHBoxLayout(); navbar.addWidget(label_text('Frame source :'))
        self.frame_slider=QSlider(Qt.Orientation.Horizontal); self.frame_slider.setRange(0,0); self.frame_slider.valueChanged.connect(self.navigator_moved)
        navbar.addWidget(self.frame_slider, stretch=1)
        self.frame_badge=label_text('Image 1', 'metric'); navbar.addWidget(self.frame_badge)
        navl.addLayout(navbar)
        pick=QHBoxLayout();pick.setSpacing(10)
        self.custom_btn=QPushButton('＋  Ajouter un masque personnalisé')
        self.custom_btn.setObjectName('outlined')
        self.custom_btn.clicked.connect(self.toggle_custom_selection)
        pick.addWidget(self.custom_btn,stretch=2)
        self.replace_slot=QComboBox();[self.replace_slot.addItem(f'Test {i+1}',i) for i in range(3)]
        pick.addWidget(self.replace_slot,stretch=1)
        self.replace_btn=QPushButton('Utiliser cette image');self.replace_btn.clicked.connect(self.replace_sample_from_navigator)
        pick.addWidget(self.replace_btn)
        self.random_btn=QPushButton('↻  Images aléatoires');self.random_btn.clicked.connect(self.randomize_samples)
        pick.addWidget(self.random_btn)
        navl.addLayout(pick)
        self.custom_hint=label_text('Sélection personnalisée : placez le curseur sur la bonne frame, puis dessinez un rectangle sur la vidéo source.')
        navl.addWidget(self.custom_hint)

        selected_row=QHBoxLayout()
        self.tracks=label_text('Aucun suivi personnalisé.');selected_row.addWidget(self.tracks,stretch=1)
        self.remove_last=QPushButton('Retirer le dernier');self.remove_last.clicked.connect(self.remove_last_selection);selected_row.addWidget(self.remove_last)
        self.clear_btn=QPushButton('Tout effacer');self.clear_btn.clicked.connect(self.clear_selections);selected_row.addWidget(self.clear_btn)
        hl.addLayout(selected_row)
        self.note=label_text('Le suivi automatique des objets sélectionnés fonctionne en avant et en arrière, mais peut décrocher sur une occultation ou un changement de plan.')
        hl.addWidget(self.note)
        bottom,bl=card('Rendu final')
        left.addWidget(bottom)
        rr=QHBoxLayout();self.source_info=label_text('Aucun fichier importé')
        rr.addWidget(self.source_info,stretch=1);bl.addLayout(rr)
        br=QHBoxLayout();self.export_btn=QPushButton('↗  Appliquer à tout le média');self.export_btn.setObjectName('primary')
        self.export_btn.clicked.connect(self.export);self.export_btn.setEnabled(False);br.addWidget(self.export_btn,stretch=1)
        self.cancel_btn=QPushButton('Annuler');self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel);br.addWidget(self.cancel_btn);bl.addLayout(br)
        self.progress=QProgressBar();self.progress.setValue(0);bl.addWidget(self.progress)
        self.status=label_text('Prêt à importer une image ou une vidéo.');bl.addWidget(self.status)
        sidebar=QScrollArea();sidebar.setWidgetResizable(True);sidebar.setFixedWidth(400)
        panel=QWidget();panel.setObjectName('surface');sidebar.setWidget(panel)
        right=QVBoxLayout(panel);right.setContentsMargins(4,1,7,4);right.setSpacing(19)
        main.addWidget(sidebar)
        general,gl=card('01   Détections automatiques','Sélectionne ce qui doit disparaître dans le résultat.')
        right.addWidget(general)
        self.scope=QComboBox();self.scope.addItem('Masquer tout le monde automatiquement','all');self.scope.addItem('Masquer seulement les sujets sélectionnés','selected');self.scope.currentIndexChanged.connect(self.change_settings);gl.addWidget(self.scope)
        self.faces=QCheckBox('Visages · SCRFD + YuNet')
        self.heads=QCheckBox('Têtes de face et de dos');self.heads.setChecked(True)
        self.persons=QCheckBox('Corps entiers');self.persons.setChecked(True)
        self.animals=QCheckBox('Animaux');self.screens=QCheckBox('Écrans et téléphones')
        self.tables=QCheckBox('Tables (classe COCO)');self.colors=QCheckBox('Couleur pipette')
        self.shapes=QCheckBox('Formes géométriques')
        for cb in (self.faces,self.heads,self.persons,self.animals,self.screens,self.tables,self.colors,self.shapes):
            gl.addWidget(cb);cb.toggled.connect(self.change_settings)
        self.faces.blockSignals(True);self.faces.setChecked(True);self.faces.blockSignals(False)
        contour,co=card('02   Forme & masquage',
            'Un même style s’applique à la tête et au corps, avec la marge de sécurité adaptée à chaque zone.')
        right.addWidget(contour)
        co.addWidget(label_text('STYLE DU CONTOUR', 'eyebrow'))
        self.contour_style=QComboBox()
        self.contour_style.addItem('▢  Rectangle complet · sécurité','box')
        self.contour_style.addItem('◌  Contour lisse · silhouette','smooth')
        self.contour_style.addItem('▦  Contour pixélisé · petits carrés','pixelated')
        self.contour_style.setCurrentIndex(2)
        self.contour_style.currentIndexChanged.connect(self.change_settings)
        co.addWidget(self.contour_style)
        co.addWidget(label_text('EFFET APPLIQUÉ', 'eyebrow'))
        self.mode=QComboBox();self.mode.addItem('Flou puissant','blur')
        self.mode.addItem('Masque noir','black');self.mode.addItem('Pixelisation','pixel')
        self.mode.currentIndexChanged.connect(self.change_settings);co.addWidget(self.mode)
        self.block_label=label_text('Taille des petits carrés : 14 px');co.addWidget(self.block_label)
        self.block=QSlider(Qt.Orientation.Horizontal);self.block.setRange(4,48);self.block.setValue(14)
        self.block.valueChanged.connect(self.change_settings);co.addWidget(self.block)
        self.head_label=label_text('Marge autour de la tête : 55 %');co.addWidget(self.head_label)
        self.head_pad=QSlider(Qt.Orientation.Horizontal);self.head_pad.setRange(10,130);self.head_pad.setValue(55)
        self.head_pad.valueChanged.connect(self.change_settings);co.addWidget(self.head_pad)
        self.body_label=label_text('Marge autour du corps : 10 %');co.addWidget(self.body_label)
        self.body_pad=QSlider(Qt.Orientation.Horizontal);self.body_pad.setRange(0,35);self.body_pad.setValue(10)
        self.body_pad.valueChanged.connect(self.change_settings);co.addWidget(self.body_pad)
        self.contour_note=label_text('Corps : segmentation YOLO11-seg. Têtes : contour elliptique estimé. Objets personnalisés : GrabCut. En cas d’échec : masque rectangulaire de repli.')
        co.addWidget(self.contour_note)
        self.fail_closed=QCheckBox('Écran noir si aucune tête détectée')
        co.addWidget(self.fail_closed);self.fail_closed.toggled.connect(self.change_settings)
        quality,ql=card('03   Qualité de détection')
        right.addWidget(quality)
        self.face_quality=QComboBox()
        for text,data in [('Extrême · tuiles / miroir / contraste','extreme'),
                          ('Précise · SCRFD + YuNet','precise'),('Rapide · YuNet','standard')]:
            self.face_quality.addItem(text,data)
        self.face_quality.currentIndexChanged.connect(self.change_settings);ql.addWidget(self.face_quality)
        self.stride=QSpinBox();self.stride.setRange(1,20);self.stride.setValue(4);self.stride.setSuffix(' images')
        self.stride.valueChanged.connect(self.change_settings)
        ql.addWidget(label_text('Intervalle des objets (visages : toutes les images)'))
        ql.addWidget(self.stride)
        self.hue=QSlider(Qt.Orientation.Horizontal);self.hue.setRange(2,30);self.hue.setValue(10)
        self.hue.valueChanged.connect(self.change_settings)
        ql.addWidget(label_text('Tolérance couleur'));ql.addWidget(self.hue)
        output,ol=card('04   Export et audio')
        right.addWidget(output)
        self.quality=QComboBox()
        for text,data in [('MP4 · haute qualité','high'),('MP4 · export rapide','fast'),
                          ('MP4 · AMD GPU (si disponible)','amd'),('MKV · FFV1 sans perte','lossless')]:
            self.quality.addItem(text,data)
        ol.addWidget(self.quality)
        self.mute=QCheckBox('Supprimer le son');self.mute.toggled.connect(self.change_settings)
        ol.addWidget(self.mute)
        self.volume_label=label_text('Volume audio : 100 %');ol.addWidget(self.volume_label)
        self.volume=QSlider(Qt.Orientation.Horizontal);self.volume.setRange(0,200);self.volume.setValue(100)
        self.volume.valueChanged.connect(self.change_settings);ol.addWidget(self.volume)
        ol.addWidget(label_text('À 100 %, piste audio copiée sans réencodage. Sinon audio AAC réencodé. Images : sans audio.'))
        models,mdl=card('Moteurs installés','Modèles pré-entraînés : aucun entraînement requis.')
        right.addWidget(models)
        self.models_label=label_text('Chargement du statut…');mdl.addWidget(self.models_label)
        right.addStretch()
        self.change_settings()

    def toggle_theme(self):
        self.dark=not self.dark;self.refresh_theme()

    def toggle_compare(self):
        self.show_original=not self.show_original
        self.compare_btn.setText('◫  Voir les masques' if self.show_original else '◫  Voir les originaux')
        self.display_samples()

    def toggle_custom_selection(self):
        if self.active or (self.export_thread is not None and self.export_thread.isRunning()):
            return
        if not self.path:
            QMessageBox.information(self, 'Importer un média', 'Importez une vidéo ou une image avant de sélectionner un objet.')
            return
        self.selecting_custom = not self.selecting_custom
        self.nav_canvas.tool = 'custom' if self.selecting_custom else 'view'
        self.nav_canvas.setCursor(Qt.CursorShape.CrossCursor if self.selecting_custom else Qt.CursorShape.PointingHandCursor)
        self.custom_btn.setText('✓  Dessinez sur l’image source…' if self.selecting_custom else '＋  Ajouter un masque personnalisé')
        self.custom_btn.setObjectName('selected' if self.selecting_custom else 'outlined')
        self.custom_btn.style().unpolish(self.custom_btn)
        self.custom_btn.style().polish(self.custom_btn)
        self.status.setText('Tracez un rectangle autour de l’objet sur la frame source.' if self.selecting_custom
                            else 'Sélection personnalisée terminée.')

    def update_models(self):
        parts=[('Visages',SCRFD_PATH.exists() and YUNET_PATH.exists()),
               ('Têtes',HEAD_MODEL_PATH.exists()),('Objets / personnes',MODEL_PATH.exists()),
               ('Silhouettes en blocs',PERSON_SEG_PATH.exists())]
        self.models_label.setText('   ·   '.join(('✓ ' if ok else '○ ')+name for name,ok in parts)+
            '\nInstaller une fois avec INSTALLER.bat si des modèles manquent.')

    def open_media(self):
        if self.active or (self.export_thread is not None and self.export_thread.isRunning()):
            QMessageBox.information(self, "Traitement en cours", "Terminez l’analyse avant de changer de média.")
            return
        path,_=QFileDialog.getOpenFileName(self,'Importer une vidéo ou une image','',
            'Images et vidéos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;Tous les fichiers (*.*)')
        if not path:return
        try:
            ext=Path(path).suffix.lower();kind='image' if ext in IMAGE_EXT else 'video'
            if kind=='image':
                frame=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
                if frame is None:raise ValueError('Image non reconnue.')
                frames=[frame];indices=[0];info=None
                desc=f'{Path(path).name}   /   Image {frame.shape[1]} × {frame.shape[0]}'
            else:
                info=inspect_video(path)
                if info.frames<1:raise ValueError('Nombre d’images vidéo non disponible.')
                indices=[0,0,0]; frames=[]
                desc=f'{Path(path).name}   /   {info.width} × {info.height}   /   {info.fps:.2f} FPS   /   {info.frames} images'
        except Exception as exc:
            QMessageBox.critical(self,'Import impossible',str(exc));return
        self.path=path;self.kind=kind;self.info=info;self.sample_frames=frames;self.sample_indices=indices
        if self.selecting_custom: self.toggle_custom_selection()
        self.random_btn.setEnabled(kind=='video'); self.frame_slider.setEnabled(kind=='video')
        self.settings.regions=[];self.settings.custom_objects=[];self.clear_btn.setEnabled(True)
        self.source_info.setText(desc);self.export_btn.setEnabled(True)
        self.preview_outputs=[]
        if kind=='image':
            self.navigator_frame=frames[0]; self.nav_canvas.show_bgr(frames[0]); self.frame_badge.setText('Image source')
            for i,canvas in enumerate(self.canvases):
                canvas.setVisible(i<1 if kind=='image' else True)
                if i<1:
                    canvas.show_bgr(frames[0]); self.sample_badges[i].setText('#1')
                else:
                    canvas.clear(); canvas.setText('—')
            self.apply_btn.setEnabled(True)
        else:
            self.frame_slider.setRange(0, max(0, info.frames-1)); self.frame_slider.setValue(0)
            self.navigator_moved(0)
            self.set_samples(self.random_sample_indices())
        self.change_settings()

    def load_frame(self, index:int):
        if self.kind != 'video' or not self.path or self.info is None:
            return None
        index=max(0,min(self.info.frames-1,index))
        cap=cv2.VideoCapture(self.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok:
                return None
            return frame
        finally:
            cap.release()

    def random_sample_indices(self):
        if self.kind != 'video' or self.info is None or self.info.frames < 1:
            return [0]
        n=min(3, self.info.frames)
        if self.info.frames <= 3:
            return list(range(self.info.frames))
        population=range(self.info.frames)
        idx=sorted(random.sample(population, n))
        while len(idx) < 3:
            idx.append(idx[-1])
        return idx

    def set_samples(self, indices:list[int]):
        if self.active or (self.export_thread is not None and self.export_thread.isRunning()):
            return
        if self.kind == 'image':
            return
        frames=[]
        for index in indices:
            frame=self.load_frame(index)
            if frame is None: raise ValueError(f'Impossible de lire l’image {index+1}.')
            frames.append(frame)
        self.sample_frames=frames; self.sample_indices=indices
        self.preview_version += 1
        self.preview_outputs=[]
        for canvas in self.canvases: canvas.setVisible(True)
        for i,(canvas,index,frame) in enumerate(zip(self.canvases,indices,frames)):
            self.sample_badges[i].setText(f'#{index+1}')
            canvas.show_bgr(frame)
        self.apply_btn.setEnabled(True)
        self.preview_outputs=[]
        self.status.setText('3 images prêtes. Clique sur “Appliquer aux 3 images”.')

    def randomize_samples(self):
        if self.kind != 'video' or self.info is None: return
        self.set_samples(self.random_sample_indices())

    def navigator_moved(self, value:int):
        self.current_frame_index=int(value)
        if self.kind != 'video' or self.info is None: return
        frame=self.load_frame(self.current_frame_index)
        if frame is None: return
        self.navigator_frame=frame
        self.nav_canvas.show_bgr(frame)
        self.frame_badge.setText(f'Image {self.current_frame_index+1} / {self.info.frames}')

    def replace_sample_from_navigator(self):
        if self.active or (self.export_thread is not None and self.export_thread.isRunning()): return
        if self.kind != 'video' or self.navigator_frame is None: return
        slot=int(self.replace_slot.currentData())
        if slot >= len(self.canvases): return
        if len(self.sample_frames) < 3:
            self.sample_frames=[self.navigator_frame.copy()]*3; self.sample_indices=[self.current_frame_index]*3
        self.preview_version += 1
        self.sample_frames[slot]=self.navigator_frame.copy()
        self.sample_indices[slot]=self.current_frame_index
        self.canvases[slot].show_bgr(self.sample_frames[slot])
        self.sample_badges[slot].setText(f'#{self.current_frame_index+1}')
        self.preview_outputs=[]
        self.display_samples()
        self.apply_btn.setEnabled(True)
        self.status.setText(f'Image test {slot+1} remplacée. Clique sur “Appliquer aux 3 images”.')

    def add_region_from_navigator(self, x,y,w,h):
        if self.active or not self.path or not self.selecting_custom: return
        self.settings.custom_objects.append(CustomObject(self.current_frame_index,(x,y,w,h), True))
        self.toggle_custom_selection()
        self.change_settings()
        if self.kind == 'video':
            self.replace_sample_from_navigator()
        self.status.setText(f'Masque personnalisé ajouté sur la frame {self.current_frame_index+1}. Teste le rendu avec « Appliquer aux 3 images ».')

    def sample_color_from_navigator(self,x,y):
        if self.navigator_frame is None: return
        self.settings.color_bgr=tuple(int(v) for v in self.navigator_frame[y,x])
        self.colors.setChecked(True); self.change_settings()

    def open_zoom(self, index:int):
        frames=self.sample_frames if self.show_original or len(self.preview_outputs)!=len(self.sample_frames) else self.preview_outputs
        if 0 <= index < len(frames):
            dlg=ZoomDialog(f'Image test {index+1}', frames[index], self)
            dlg.exec()

    def current_settings(self):
        return replace(self.settings,
            faces=self.faces.isChecked(),heads=self.heads.isChecked(),persons=self.persons.isChecked(),
            animals=self.animals.isChecked(),screens=self.screens.isChecked(),tables=self.tables.isChecked(),
            color=self.colors.isChecked(),shapes=self.shapes.isChecked(),
            body_shape={'box':'box','smooth':'silhouette','pixelated':'blocks'}[self.contour_style.currentData()],
            body_silhouette=self.contour_style.currentData()!='box',block_size=self.block.value(),
            body_padding=self.body_pad.value()/100,head_padding=self.head_pad.value()/100,
            fail_closed=self.fail_closed.isChecked(),mode=self.mode.currentData(),
            face_quality=self.face_quality.currentData(),inference_stride=self.stride.value(),
            hue_tolerance=self.hue.value(),regions=list(self.settings.regions),
            custom_objects=list(self.settings.custom_objects), head_shape={'box':'box','smooth':'smooth','pixelated':'blocks'}[self.contour_style.currentData()],
            only_selected_subjects=self.scope.currentData()=='selected')

    @staticmethod
    def signature(cfg):
        return (cfg.faces,cfg.heads,cfg.persons,cfg.body_silhouette,cfg.body_shape, cfg.head_shape, cfg.only_selected_subjects,
                frozenset(cfg.yolo_classes),cfg.face_quality)

    def change_settings(self,*_):
        # Called during UI construction too; don't render until debounce exists.
        self.settings=self.current_settings()
        self.block_label.setText(f'Taille des carrés : {self.block.value()} px')
        self.body_label.setText(f'Marge corps : {self.body_pad.value()} %')
        self.head_label.setText(f'Marge tête : {self.head_pad.value()} %')
        self.volume_label.setText(f'Volume audio : {self.volume.value()} %')
        self.volume.setEnabled(not self.mute.isChecked())
        self.tracks.setText(f'{len(self.settings.custom_objects)} sujet(s) sélectionné(s) pour un suivi vidéo')
        self.block_label.setVisible(self.contour_style.currentData()=='pixelated')
        self.block.setVisible(self.contour_style.currentData()=='pixelated')
        self.preview_version+=1
        self.preview_outputs=[]
        self.display_samples()
        self.apply_btn.setEnabled(bool(self.sample_frames))
        self.status.setText('Réglages modifiés · clique sur “Appliquer aux 3 images”.')

    def start_preview(self):
        if not self.sample_frames or self.active:return
        if self.preview_thread is not None and self.preview_thread.isRunning():return
        generation=self.preview_version;cfg=self.current_settings()
        processor=self.cached_processor if self.cached_signature==self.signature(cfg) else None
        self.cached_processor=None
        thread=PreviewThread(generation,self.sample_frames,self.sample_indices,cfg,processor)
        self.preview_thread=thread;self.active=True
        self.progress.setRange(0,0); self.progress.setFormat('Chargement des modèles…')
        self.status.setText('Calcul en cours…')
        thread.progressChanged.connect(self.preview_progress)
        thread.ready.connect(self.preview_ready);thread.failed.connect(self.preview_failed)
        thread.finished.connect(self.preview_finished)
        thread.start()

    def preview_progress(self, value:int):
        self.progress.setRange(0,100)
        self.progress.setValue(value)
        self.progress.setFormat('%p% · images de contrôle')

    def preview_ready(self,generation,outputs,processor,_):
        self.cached_signature=self.signature(processor.settings)
        self.cached_processor=processor
        if generation != self.preview_version:return
        self.preview_outputs=outputs
        self.status.setText(f'{len(outputs)} image(s) test analysée(s). Vérifie puis exporte quand le résultat te convient.')
        self.display_samples()

    def preview_failed(self,generation,message):
        if generation!=self.preview_version:return
        self.preview_outputs=[];self.display_samples()
        self.status.setText('Aperçu IA impossible : '+message)

    def preview_finished(self):
        self.active=False
        self.progress.setRange(0,100); self.progress.setFormat('%p%')

    def display_samples(self):
        frames=self.sample_frames if self.show_original or len(self.preview_outputs)!=len(self.sample_frames) else self.preview_outputs
        for i,frame in enumerate(frames):self.canvases[i].show_bgr(frame)

    def remove_last_selection(self):
        if self.settings.custom_objects:
            self.settings.custom_objects.pop()
        self.change_settings()

    def clear_selections(self):
        self.settings.custom_objects.clear();self.settings.regions.clear();self.change_settings()

    def missing_models(self,cfg):
        if cfg.only_selected_subjects:
            # Le suivi manuel fonctionne sans modèles IA pour les personnes.
            if cfg.yolo_classes and not MODEL_PATH.exists():
                return 'Détecteur d’objets manquant : INSTALLER.bat.'
            return ''
        if cfg.faces and not YUNET_PATH.exists():return 'YuNet manquant : installer les modèles pré-entraînés.'
        if cfg.faces and cfg.face_quality!='standard' and not SCRFD_PATH.exists():return 'SCRFD manquant : installer les modèles pré-entraînés.'
        if (cfg.persons or cfg.yolo_classes) and not MODEL_PATH.exists():return 'Détecteur personnes/objets manquant : INSTALLER.bat.'
        if cfg.persons and cfg.body_shape!='box' and not PERSON_SEG_PATH.exists():return 'Segmentation YOLO11-seg manquante : INSTALLER.bat ou choisir rectangle.'
        if cfg.heads and not (HEAD_MODEL_PATH.exists() or MODEL_PATH.exists()):return 'Détecteur de tête manquant : INSTALLER.bat.'
        return ''

    def export(self):
        if not self.path:return
        cfg=self.current_settings()
        if cfg.only_selected_subjects and not cfg.custom_objects:
            QMessageBox.warning(self,'Aucun sujet choisi','Sélectionnez au moins un sujet avec « Ajouter un masque personnalisé ».')
            return
        missing=self.missing_models(cfg)
        if missing:QMessageBox.warning(self,'Modèle indisponible',missing);return
        if self.kind=='image':
            default=str(Path(self.path).with_name(Path(self.path).stem+'_masque.png'))
            destination,_=QFileDialog.getSaveFileName(self,'Exporter une image',default,'PNG (*.png);;JPEG (*.jpg);;WebP (*.webp)')
            if destination and Path(destination).suffix.lower() not in IMAGE_EXT:destination+='.png'
        else:
            extension='.mkv' if self.quality.currentData()=='lossless' else '.mp4'
            default=str(Path(self.path).with_name(Path(self.path).stem+'_masque'+extension))
            destination,_=QFileDialog.getSaveFileName(self,'Exporter la vidéo',default,
                          'MKV (*.mkv)' if extension=='.mkv' else 'MP4 (*.mp4)')
            if destination and Path(destination).suffix.lower()!=extension:destination+=extension
        if not destination:return
        if Path(destination).resolve()==Path(self.path).resolve():
            QMessageBox.warning(self,'Fichier protégé','Ne pas écraser le média source.');return
        volume=0. if self.mute.isChecked() else self.volume.value()/100
        self.export_thread=ExportThread(self.path,destination,cfg,self.quality.currentData(),self.kind,volume)
        self.export_thread.progressChanged.connect(self.progress.setValue)
        self.export_thread.completed.connect(self.export_done)
        self.export_thread.failed.connect(self.export_failed)
        self.export_thread.canceled.connect(lambda:self.status.setText('Export annulé.'))
        self.export_thread.finished.connect(self.export_unlocked)
        self.export_btn.setEnabled(False);self.open_btn.setEnabled(False);self.cancel_btn.setEnabled(True)
        self.progress.setValue(0);self.status.setText('Analyse et export local en cours…')
        self.export_thread.start()

    def export_done(self,path,count,provider):
        self.progress.setValue(100)
        self.status.setText(f'Terminé · {count} image(s) · {Path(path).name}')
        QMessageBox.information(self,'Export terminé',f'Fichier : {path}\n{count} image(s) traitée(s)\n{provider}')

    def export_failed(self,message):
        self.status.setText('Échec : '+message)
        QMessageBox.critical(self,'Échec de traitement',message)

    def export_unlocked(self):
        self.export_btn.setEnabled(bool(self.path));self.open_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def cancel(self):
        if self.export_thread:self.export_thread.abort=True

    def closeEvent(self,event):
        if self.export_thread is not None and self.export_thread.isRunning():
            QMessageBox.warning(self,'Traitement en cours','Annule l’export avant de fermer.')
            event.ignore();return
        if self.preview_thread is not None and self.preview_thread.isRunning():
            self.preview_thread.wait()  # pas de QThread détruit prématurément
        super().closeEvent(event)


def main():
    app=QApplication(sys.argv);app.setStyle('Fusion')
    app.setApplicationName('VideoMask AI · Contour Studio')
    w=MainWindow();w.show()
    return app.exec()
