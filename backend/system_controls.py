"""Windows volume; legacy screen brightness is read-only. Media brightness lives in engine."""
import ctypes as ct
import json
import subprocess
import sys
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix='/api/system')

# Read-only, shared sampling. Requests never block on a hardware counter and the
# sampler stops automatically when no performance panel has requested it.
_metrics_lock = __import__('threading').Lock()
_metrics = {'ready': False}
_metrics_thread = None
_metrics_requested = 0.

def _sample_performance():
    import time, psutil
    from performance import GpuCounter
    from hardware_status import ThermalCounter, power_status
    global _metrics
    gpu = GpuCounter();thermal = ThermalCounter()
    psutil.cpu_percent()
    try:
        while time.monotonic() - _metrics_requested < 15:
            time.sleep(1)
            memory = psutil.virtual_memory()
            value = {'ready': True, 'sampled_at': time.time(),
                     'cpu_percent': psutil.cpu_percent(),
                     'ram_percent': memory.percent,
                     'ram_used_mb': (memory.total-memory.available)/1024**2,
                     'ram_total_mb': memory.total/1024**2,
                     'gpu_engine_percent': gpu.read(system=True),
                     'temperature': thermal.read(), 'power': power_status()}
            with _metrics_lock:
                _metrics = value
    finally:
        gpu.close();thermal.close()

@router.get('/performance')
def get_performance():
    import threading, time
    global _metrics_thread, _metrics_requested
    with _metrics_lock:
        _metrics_requested = time.monotonic()
        if _metrics_thread is None or not _metrics_thread.is_alive():
            _metrics_thread = threading.Thread(target=_sample_performance, daemon=True)
            _metrics_thread.start()
        return dict(_metrics)

class Level(BaseModel):
    value: int = Field(ge=0, le=100)

class GUID(ct.Structure):
    _fields_ = [('data', ct.c_ubyte * 16)]
    def __init__(self, value):
        super().__init__()
        self.data[:] = uuid.UUID(value).bytes_le

def endpoint_volume(value=None, with_mute=False):
    if sys.platform != 'win32':
        raise RuntimeError('Réglage disponible sur Windows uniquement.')
    ole = ct.OleDLL('ole32')
    ole.CoInitializeEx(None, 0)
    pointers = []
    def call(ptr, index, args, *values):
        table = ct.cast(ptr, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
        result = ct.WINFUNCTYPE(ct.c_long, ct.c_void_p, *args)(table[index])(ptr, *values)
        if result < 0:
            raise RuntimeError('Aucune sortie audio Windows disponible.')
    try:
        enumerator = ct.c_void_p()
        ole.CoCreateInstance(ct.byref(GUID('BCDE0395-E52F-467C-8E3D-C4579291692E')), None, 1,
                             ct.byref(GUID('A95664D2-9614-4F35-A746-DE8DB63617E6')), ct.byref(enumerator))
        pointers.append(enumerator)
        device = ct.c_void_p()
        call(enumerator, 4, [ct.c_int, ct.c_int, ct.POINTER(ct.c_void_p)], 0, 1, ct.byref(device))
        pointers.append(device)
        endpoint = ct.c_void_p()
        call(device, 3, [ct.POINTER(GUID), ct.c_uint, ct.c_void_p, ct.POINTER(ct.c_void_p)],
             ct.byref(GUID('5CDF2C82-841E-4546-9722-0CF74078229A')), 23, None, ct.byref(endpoint))
        pointers.append(endpoint)
        if value is not None:
            call(endpoint, 7, [ct.c_float, ct.c_void_p], value / 100, None)
            call(endpoint, 14, [ct.c_int, ct.c_void_p], int(value == 0), None)
        current = ct.c_float()
        call(endpoint, 9, [ct.POINTER(ct.c_float)], ct.byref(current))
        muted = ct.c_int()
        call(endpoint, 15, [ct.POINTER(ct.c_int)], ct.byref(muted))
        level = round(current.value * 100)
        return {'value': level, 'muted': bool(muted.value)} if with_mute else level
    finally:
        for pointer in reversed(pointers):
            if pointer.value:
                table = ct.cast(pointer, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
                ct.WINFUNCTYPE(ct.c_ulong, ct.c_void_p)(table[2])(pointer)
        ole.CoUninitialize()

def brightness(value=None):
    if sys.platform != 'win32':
        raise RuntimeError('Réglage disponible sur Windows uniquement.')
    # Only validated integer values enter this fixed PowerShell program.
    command = "$ErrorActionPreference='Stop'; "
    if value is not None:
        command += f"Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods | ForEach-Object {{Invoke-CimMethod -InputObject $_ -MethodName WmiSetBrightness -Arguments @{{Timeout=[uint32]0;Brightness=[byte]{value}}} | Out-Null}}; "
    command += "@(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness | Where-Object Active | Select-Object -ExpandProperty CurrentBrightness) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, timeout=8, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError('Cet écran ne fournit pas de contrôle de luminosité Windows (WMI).')
    levels = json.loads(result.stdout)
    if not isinstance(levels, list):
        levels = [levels]
    if not levels:
        raise RuntimeError('Aucun écran compatible avec la luminosité Windows.')
    return int(levels[0])

def capability(function):
    try:
        return {'available': True, 'value': function(), 'reason': ''}
    except Exception as error:
        return {'available': False, 'value': 0, 'reason': str(error)}

@router.get('/volume')
def get_volume():
    try:
        return {'available': True, **endpoint_volume(with_mute=True), 'reason': ''}
    except Exception as error:
        return {'available': False, 'value': 0, 'muted': False, 'reason': str(error)}

@router.get('')
def get_levels():
    return {'volume': get_volume(), 'brightness': capability(brightness)}

@router.post('/{control}')
def set_level(control: str, body: Level):
    if control == 'brightness':
        raise HTTPException(410, 'La luminosité se règle désormais sur le média dans VideoMask.')
    if control not in ('volume',):
        raise HTTPException(404, 'Réglage inconnu.')
    try:
        value = endpoint_volume(body.value, with_mute=True)
        return {'available': True, **value, 'reason': ''}
    except Exception as error:
        raise HTTPException(409, str(error)) from error
