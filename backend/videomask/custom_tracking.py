"""Suivi bidirectionnel d'un objet choisi manuellement (CSRT ou template matching).

Confiance insuffisante -> maintien temporaire, puis arrêt du suivi. Pas de
ré-identification garantie après occultation ou changement de scène.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

Box=tuple[int,int,int,int]

@dataclass(frozen=True)
class CustomObject:
    frame: int
    box: Box
    contour: bool = False


def _create_csrt():
    ctor=getattr(cv2,'TrackerCSRT_create',None)
    if ctor is None:
        ctor=getattr(getattr(cv2,'legacy',None),'TrackerCSRT_create',None)
    return ctor() if ctor else None


def _gray(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

class ObjectFollower:
    def __init__(self,frame:np.ndarray,box:Box):
        self.box=tuple(int(i) for i in box)
        self.gray=_gray(frame)
        self.template=self._crop(self.gray,self.box)
        self.misses=0
        self.confidence=1.
        self.csrt=_create_csrt()
        if self.csrt is not None:
            try:
                self.csrt.init(frame,self.box)
            except (cv2.error,AttributeError):
                self.csrt=None

    @staticmethod
    def _crop(im,box):
        x,y,w,h=box
        return im[max(0,y):max(0,y+h),max(0,x):max(0,x+w)].copy()

    def update(self,frame:np.ndarray)->Box|None:
        gray=_gray(frame)
        if self.csrt is not None:
            try:
                ok,b=self.csrt.update(frame)
                if ok:
                    x,y,w,h=(int(round(v)) for v in b)
                    if w>4 and h>4:
                        patch=self._crop(gray,(x,y,w,h))
                        if patch.size and self.template.std()>8:
                            patch=cv2.resize(patch,(self.template.shape[1],self.template.shape[0]))
                            similarity=float(cv2.matchTemplate(patch,self.template,cv2.TM_CCOEFF_NORMED)[0,0])
                            self.confidence=max(0.,similarity)
                            if similarity>=.18:
                                self.box=(x,y,w,h);self.misses=0
                                self.template=cv2.addWeighted(self.template,.9,patch,.1,0)
                                return self.box
                        self.csrt=None
            except cv2.error:
                self.csrt=None
        x,y,w,h=self.box
        th,tw=self.template.shape[:2]
        if tw<5 or th<5 or self.template.std()<8: return None
        margin=max(18,int(max(w,h)*.7))
        xa=max(0,x-margin);ya=max(0,y-margin)
        xb=min(gray.shape[1],x+w+margin);yb=min(gray.shape[0],y+h+margin)
        search=gray[ya:yb,xa:xb]
        if search.shape[0]>=th and search.shape[1]>=tw:
            scores=cv2.matchTemplate(search,self.template,cv2.TM_CCOEFF_NORMED)
            _,score,_,loc=cv2.minMaxLoc(scores)
            self.confidence=max(0.,float(score))
            if score>=.46:
                self.box=(xa+loc[0],ya+loc[1],w,h)
                self.misses=0
                return self.box
        self.misses+=1
        self.confidence=0.
        return self.box if self.misses<=5 else None


def plan_custom_objects(path:str, objects:list[CustomObject], total:int,
                        progress=None,cancel=None)->list[list[tuple[Box,bool]]]:
    """Parcours avant et arrière de chaque graine. Mémoire O(nb_images * objets).

    Les objets sont suivis indépendamment ; des chevauchements sont possibles.
    """
    plan: list[list[tuple[Box,bool]]] = [[] for _ in range(total)]
    if not objects or total<1: return plan
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError('Vidéo illisible pour le suivi manuel.')
    try:
        for oi,obj in enumerate(objects):
            if cancel and cancel(): raise InterruptedError('Analyse annulée.')
            anchor=max(0,min(total-1,obj.frame))
            cap.set(cv2.CAP_PROP_POS_FRAMES,anchor)
            ok,first=cap.read()
            if not ok: raise ValueError(f'Image de sélection {anchor} illisible.')
            plan[anchor].append((obj.box,obj.contour))
            for step in (1,-1):
                follower=ObjectFollower(first,obj.box)
                i=anchor+step
                while 0<=i<total:
                    if cancel and cancel(): raise InterruptedError('Analyse annulée.')
                    if step<0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
                    ok,frame=cap.read()
                    if not ok: break
                    box=follower.update(frame)
                    if box is None: break
                    plan[i].append((box,obj.contour))
                    i+=step
            if progress: progress(round(100*(oi+1)/len(objects)))
    finally:
        cap.release()
    return plan
