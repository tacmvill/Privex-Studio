"""Thread-scoped Windows idle-sleep inhibition for local processing only."""
import ctypes,sys

class KeepAwake:
 def __init__(self,api=None):self.api=api;self.active=False;self.error=''
 def acquire(self):
  if self.active:return True
  if self.api is None:
   if sys.platform!='win32':return False
   self.api=ctypes.windll.kernel32.SetThreadExecutionState
   self.api.argtypes=[ctypes.c_uint];self.api.restype=ctypes.c_uint
  try:
   self.active=bool(self.api(0x80000001)) # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
   if not self.active:self.error='Windows a refusé la prévention de veille.'
  except Exception as exc:self.error=str(exc)
  return self.active
 def release(self):
  if self.active:
   try:self.api(0x80000000)
   finally:self.active=False
