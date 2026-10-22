"""Atomic plan persistence and indexed frame access for responsive timeline previews."""
from pathlib import Path
import json,os,sqlite3,time
from storage_paths import DATA
def replace(source,target):
 for _ in range(100):
  try:os.replace(source,target);return
  except PermissionError:time.sleep(.01)
 raise PermissionError('Le cache de suivi est occupé. Réessayez.')
def save(key,plan):
 __import__('render_continuity').repair(plan)
 path=DATA/(key+'.frames.sqlite3');tmp=path.with_suffix('.tmp')
 if tmp.exists():tmp.unlink()
 tmp.with_name(tmp.name+'-journal').unlink(missing_ok=True)
 connection=sqlite3.connect(tmp)
 try:
  connection.execute('CREATE TABLE frames (frame INTEGER PRIMARY KEY, objects TEXT NOT NULL)')
  connection.executemany('INSERT INTO frames VALUES (?,?)',((i,json.dumps(row,ensure_ascii=False)) for i,row in enumerate(plan['frames'])))
  connection.commit()
 finally:connection.close()
 replace(tmp,path)
 for suffix,value in [('.analysis.json',plan.get('analysis',{})),('.catalog.json',plan.get('catalog',[])),('.plan.json',plan)]:
  path=DATA/(key+suffix);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');replace(tmp,path)

def metadata(key):
 path=DATA/(key+'.analysis.json')
 return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
def frame(key,index):
 path=DATA/(key+'.frames.sqlite3')
 if not path.exists():return None
 connection=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
 try:row=connection.execute('SELECT objects FROM frames WHERE frame=?',(index,)).fetchone();return json.loads(row[0]) if row else None
 finally:connection.close()
def catalog(key):
 path=DATA/(key+'.catalog.json')
 if path.exists():return json.loads(path.read_text(encoding='utf-8'))
 return []

def window(key,first,last):
 path=DATA/(key+'.frames.sqlite3')
 if not path.exists():return None
 connection=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
 try:return [(i,json.loads(objects)) for i,objects in connection.execute('SELECT frame,objects FROM frames WHERE frame>=? AND frame<? ORDER BY frame',(first,last))]
 finally:connection.close()
