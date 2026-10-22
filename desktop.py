"""WebView2 host: the title-bar cross uses the app's Save / Discard / Cancel dialog."""
from pathlib import Path
import threading,webbrowser,traceback

class CloseBridge:
 def __init__(self):self._window=None;self._allowed=False;self._fullscreen=False
 def toggle_fullscreen(self):
  if self._window:self._window.toggle_fullscreen();self._fullscreen=not self._fullscreen
  return self._fullscreen
 def finish_close(self):
  self._allowed=True
  if self._window:self._window.destroy()
  return True
 def closing(self):
  if self._allowed:return True
  # Never block the GUI closing event on JavaScript evaluation: it needs that GUI thread.
  threading.Thread(target=self._request,daemon=True).start()
  return False
 def _request(self):
  try:
   ready=self._window.evaluate_js("Boolean(document.querySelector('[data-studio-ready]'))")
   if ready:self._window.run_js("window.dispatchEvent(new Event('videomask:close-request'))")
   elif self._window.create_confirmation_dialog('Fermer Privex Studio','L’interface n’est pas prête. Fermer la fenêtre ?'):self.finish_close()
  except Exception:
   if self._window.create_confirmation_dialog('Fermer Privex Studio','Impossible de vérifier le projet. Fermer la fenêtre ?'):self.finish_close()

def main(url='http://127.0.0.1:8765/',storage_path=None,title='Privex Studio'):
 import webview
 webview.settings['ALLOW_DOWNLOADS']=True
 import clr
 clr.AddReference('System.Windows.Forms')
 from System.Windows.Forms import Screen
 area=Screen.PrimaryScreen.WorkingArea
 width=min(1440,max(980,area.Width-48));height=min(900,max(600,area.Height-48))
 bridge=CloseBridge()
 window=webview.create_window(title,url,width=width,height=height,min_size=(980,600),background_color='#1c1d21')
 bridge._window=window;window.events.closing+=bridge.closing
 # Expose only the two callable commands. Passing the bridge as js_api made
 # pywebview traverse the native WinForms/COM window recursively at startup.
 window.expose(bridge.toggle_fullscreen,bridge.finish_close)
 webview.start(gui='edgechromium',private_mode=False,storage_path=str(storage_path or Path(__file__).resolve().parent/'backend/data/webview'))
if __name__=='__main__':
 try:main()
 except Exception:
  path=Path(__file__).resolve().parent/'backend/data/desktop.log';path.write_text(traceback.format_exc(),encoding='utf-8')
  webbrowser.open('http://127.0.0.1:8765/')
