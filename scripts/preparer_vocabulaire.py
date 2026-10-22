from pathlib import Path
from ultralytics import YOLOE
import json,shutil
root=Path(__file__).resolve().parents[1]/'backend/models'
names=['kick scooter','trash bin','street sign','speed limit sign','Paris street name sign','wristwatch','digital clock','Eiffel Tower','Arc de Triomphe','Montparnasse Tower','Nike logo','Apple logo','Orange telecom logo','Renault logo','Ferrari logo','Nutella logo','Cristaline logo','Acer logo','Samsung logo','Ralph Lauren logo','Lacoste logo']
labels=['trottinette','poubelle','panneau de rue','limitation de vitesse','panneau de Paris','montre','heure numérique','tour Eiffel','arc de triomphe','tour Montparnasse','marque Nike','marque Apple','marque Orange','marque Renault','marque Ferrari','marque Nutella','marque Cristaline','marque Acer','marque Samsung','marque Ralph Lauren','marque Lacoste']
model=YOLOE('yoloe-11s-seg.pt')
model.set_classes(names,model.get_text_pe(names))
out=model.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False)
shutil.copy2(out,root/'open-vocabulary.onnx')
(root/'open-vocabulary.json').write_text(json.dumps(dict(prompts=names,labels=labels),ensure_ascii=False),encoding='utf-8')
print('OPEN_VOCABULARY_READY',flush=True)
