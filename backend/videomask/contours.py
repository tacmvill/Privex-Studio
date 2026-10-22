"""Masques de corps en mosaïque rectangulaire, par contours ou boîte complète.

Le modèle de segmentation produit le masque. Les blocs remplis suivent le contour
à résolution de grille ; il ne s'agit pas d'une segmentation faite par rectangles.
"""
from __future__ import annotations
import cv2
import numpy as np


def tiled_silhouette(mask: np.ndarray, block: int = 14, dilation: int = 5) -> np.ndarray:
    """Recouvre toute cellule contenant un pixel sujet (évite les trous)."""
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError('Masque mono-canal uint8 requis.')
    block = max(3, min(96, int(block)))
    if not np.any(mask):
        return np.zeros_like(mask)
    base = mask > 0
    if dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation+1, 2*dilation+1))
        base = cv2.dilate(base.astype(np.uint8), k) > 0
    h,w = base.shape
    hh = (h+block-1)//block
    ww = (w+block-1)//block
    # Aire par bloc, y compris les bordures partiellement remplies.
    padded = np.pad(base, ((0,hh*block-h),(0,ww*block-w)))
    covered = padded.reshape(hh,block,ww,block).any(axis=(1,3))
    return (np.repeat(np.repeat(covered,block,axis=0),block,axis=1)[:h,:w].astype(np.uint8)*255)


def grabcut_selected(frame: np.ndarray, box: tuple[int,int,int,int], margin: int = 5) -> np.ndarray:
    """Segmentation locale d'un sujet librement choisi, sans entraînement.

    Repli prudent sur toute la boîte si GrabCut échoue / isole trop peu de pixels.
    """
    h,w = frame.shape[:2]
    x,y,bw,bh = map(int,box)
    x=max(0,x);y=max(0,y);bw=max(0,min(w-x,bw));bh=max(0,min(h-y,bh))
    result=np.zeros((h,w),np.uint8)
    if bw<8 or bh<8:
        return result
    result[y:y+bh,x:x+bw]=255
    pad=max(10,round(max(bw,bh)*.20))
    xa,ya=max(0,x-pad),max(0,y-pad)
    xb,yb=min(w,x+bw+pad),min(h,y+bh+pad)
    region=frame[ya:yb,xa:xb]
    if min(region.shape[:2])<12:
        return result
    rx,ry=x-xa,y-ya
    box_roi=(rx,ry,bw,bh)
    try:
        m=np.zeros(region.shape[:2],np.uint8)
        bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
        cv2.grabCut(region,m,box_roi,bg,fg,2,cv2.GC_INIT_WITH_RECT)
        selected=np.where((m==cv2.GC_FGD)|(m==cv2.GC_PR_FGD),255,0).astype(np.uint8)
        coverage=np.count_nonzero(selected[ry:ry+bh,rx:rx+bw])/max(1,bw*bh)
        if coverage < .18:
            return result
        new=np.zeros((h,w),np.uint8)
        new[ya:yb,xa:xb]=selected
        if margin>0:
            k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*margin+1,2*margin+1))
            new=cv2.dilate(new,k)
        return new
    except cv2.error:
        return result
