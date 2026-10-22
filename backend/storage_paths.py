"""Writable application storage, independent of the installation directory."""
import os
from pathlib import Path

DATA = Path(os.environ.get('PRIVEX_DATA_DIR', Path(__file__).resolve().parent / 'data'))
DATA.mkdir(parents=True, exist_ok=True)
