"""Privex Studio Windows entry point and isolated processing worker."""
from pathlib import Path
import argparse
import ctypes
import logging
import multiprocessing
import os
import socket
import sys
import threading
import time

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT)]
DATA = Path(os.environ.setdefault('PRIVEX_DATA_DIR', str(Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'PrivexStudio')))
DATA.mkdir(parents=True, exist_ok=True)
IS_WORKER = '--worker' in sys.argv
for stream_name in ('stdout', 'stderr'):
    if getattr(sys, stream_name) is None:
        setattr(sys, stream_name, open(DATA / 'runtime.log', 'a', encoding='utf-8', buffering=1))


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='Privex Studio')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--serve', action='store_true', help='Serve the local interface without opening a window')
    parser.add_argument('--port', type=int, default=0, help='Local port (0 selects a free port)')
    args = parser.parse_args()
    if args.worker:
        sys.argv = ['task_worker.py', str(args.worker)]
        from task_worker import main as worker_main
        worker_main()
        return

    logging.basicConfig(filename=DATA / 'application.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
    # Windowed executables have no console streams; libraries may still write to them.
    for name in ('stdout', 'stderr'):
        if getattr(sys, name) is None:
            setattr(sys, name, open(DATA / 'runtime.log', 'a', encoding='utf-8', buffering=1))
    import uvicorn
    from app import app, JOBS, PROCESSES, _kill_task
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', args.port))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_config=None))
    if args.serve:
        print(f'http://127.0.0.1:{port}/', flush=True)
        server.run(sockets=[listener])
        return
    thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError('Le serveur local ne démarre pas. Consultez application.log.')
            time.sleep(.05)
        from desktop import main as open_window
        open_window(f'http://127.0.0.1:{port}/', DATA / 'webview')
    finally:
        for job in list(JOBS.values()):
            job['cancel'] = True
        for process in list(PROCESSES.values()):
            if process.poll() is None:
                _kill_task(process)
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        detail = traceback.format_exc()
        (DATA / 'startup-error.log').write_text(detail, encoding='utf-8')
        if not IS_WORKER and '--serve' not in sys.argv:
            ctypes.windll.user32.MessageBoxW(None,
                f'Impossible de lancer Privex Studio.\n\nVérifiez la présence de Microsoft Edge WebView2.\nJournal : {DATA / "startup-error.log"}',
                'Privex Studio', 0x10)
        else:
            print(detail, file=sys.stderr)
        sys.exit(1)
