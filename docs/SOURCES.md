# Sources et composants

Inventaire du projet étudiant. Les licences des composants s’appliquent séparément. Les liens pointent vers les projets d’origine ; les modèles binaires fournis sont également inventoriés avec leur empreinte dans `backend/models/manifest-release.json`.

## Détection et suivi

| Composant | Utilisation | Source / conditions signalées par le fournisseur |
| --- | --- | --- |
| Ultralytics YOLO11 / YOLO26 / YOLOE | Objets, personnes, segmentation et catégories supplémentaires | [Ultralytics](https://github.com/ultralytics/ultralytics), [licences](https://www.ultralytics.com/license) : AGPL-3.0 / offre Enterprise |
| YuNet | Détection de visages | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), MIT |
| SCRFD, `det_10g.onnx` | Détection de visages | [InsightFace](https://github.com/deepinsight/insightface) : code MIT, modèles préentraînés réservés à la recherche non commerciale selon le fournisseur |
| Détecteur de têtes HollywoodHeads | Têtes, y compris certains profils / dos | [abhiWanKenobi](https://huggingface.co/abhiWanKenobi/yolov8n_head_detection), CC BY-NC 4.0 |
| YOLOX-S ALPR | Plaques d’immatriculation | [Autolane](https://huggingface.co/autolane/yolox-s-alpr), Apache-2.0 ; entraînement sur plaques américaines, couverture variable ailleurs |
| Trackers | ByteTrack, OC-SORT | [Roboflow Trackers](https://github.com/roboflow/trackers), Apache-2.0 |
| Supervision | Structures et traitements de détection | [Supervision](https://github.com/roboflow/supervision), MIT |
| RapidOCR | Reconnaissance de texte | [RapidOCR](https://github.com/RapidAI/RapidOCR), Apache-2.0 ; consulter également les notices des modèles embarqués |

Extensions optionnelles depuis les sources : [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny), [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-tiny), PyTorch, Transformers et BoT-SORT / ReID. Les poids avancés et les moteurs PyTorch ne sont pas fournis dans l’application Windows standard.

## Exécution et interface

- [OpenCV](https://github.com/opencv/opencv) : vision et décodage, Apache-2.0.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) : inférence DirectML / CPU, MIT.
- [FFmpeg](https://ffmpeg.org/legal.html) et [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) : encodage ; FFmpeg conserve sa propre licence selon le binaire, le wrapper est BSD.
- [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn) et Pydantic : API locale et validation.
- [pywebview](https://github.com/r0x0r/pywebview) : fenêtre Windows / WebView2, BSD-3-Clause ; notice conservée dans `docs/licenses`.
- React, Vite, Motion et figma-squircle : interface et interactions, notices des paquets npm.
- [Lucide](https://github.com/lucide-icons/lucide) : icônes complémentaires, ISC et notices des éléments hérités de Feather.
- Polices, SF Symbols et kit Apple : voir [RESSOURCES-APPLE.md](RESSOURCES-APPLE.md).

Les versions JavaScript sont verrouillées dans `frontend/package-lock.json`. Les versions Python directes sont indiquées dans `requirements.txt` et le script d’installation. Les notices tierces accompagnant les dépendances installées doivent rester avec les distributions concernées.

## Médias et identité

Le logo Privex Studio et les ressources graphiques ont été fournis par le propriétaire du projet. Les vidéos de manifestations, captures du logiciel, musiques et voix utilisées dans la publicité sont des médias séparés : leur inclusion dans une vidéo n’accorde aucun droit de redistribution de leurs fichiers sources. Ils ne sont pas incorporés au dépôt de l’application.
