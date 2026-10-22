"""Read-only Windows power status and thermal-zone counters. No driver installation."""
import ctypes as c
from ctypes import wintypes as w
import math
import sys

class PowerStatus(c.Structure):
    _fields_ = [('ac', w.BYTE), ('battery', w.BYTE), ('percent', w.BYTE),
                ('saver', w.BYTE), ('remaining', w.DWORD), ('full', w.DWORD)]

def decode_power(status):
    present = None if status.battery == 255 else not bool(status.battery & 128)
    source = 'ac' if status.ac == 1 else 'battery' if status.ac == 0 and present is not False else 'unknown'
    return {'available': source != 'unknown', 'source': source, 'battery_present': present,
            'battery_percent': int(status.percent) if present is not False and status.percent <= 100 else None,
            'charging': bool(status.battery & 8) if present and status.battery != 255 else False,
            'battery_saver': status.saver == 1}

def power_status():
    try:
        if sys.platform != 'win32':raise OSError('Windows only')
        fn = c.WinDLL('kernel32', use_last_error=True).GetSystemPowerStatus
        fn.argtypes = [c.POINTER(PowerStatus)];fn.restype = w.BOOL
        status = PowerStatus()
        if not fn(c.byref(status)):raise OSError('Power status unavailable')
        return decode_power(status)
    except (OSError, AttributeError):
        return {'available': False, 'source': 'unknown', 'battery_present': None,
                'battery_percent': None, 'charging': False, 'battery_saver': None}

def thermal_sample(values):
    sensors = []
    for name, kelvin, status in values:
        if status not in (0, 1) or not math.isfinite(kelvin):continue
        value = kelvin - 273.15
        if not -20 <= value <= 150:continue
        sensors.append({'id': name, 'label': 'Zone thermique Windows', 'celsius': round(value, 1)})
    hottest = max(sensors, key=lambda s: s['celsius']) if sensors else None
    return {'available': bool(sensors), 'celsius': hottest['celsius'] if hottest else None,
            'sensor_id': hottest['id'] if hottest else None, 'label': 'Zone thermique Windows',
            'source': 'Windows PDH / firmware', 'sensors': sensors}

class ThermalCounter:
    def __init__(self):
        self.query = None
        try:
            if sys.platform != 'win32':return
            self.p = c.WinDLL('pdh');self.query = c.c_void_p();self.counter = c.c_void_p()
            self.p.PdhOpenQueryW.argtypes = [w.LPCWSTR, c.c_size_t, c.POINTER(c.c_void_p)]
            self.p.PdhAddEnglishCounterW.argtypes = [c.c_void_p, w.LPCWSTR, c.c_size_t, c.POINTER(c.c_void_p)]
            self.p.PdhCollectQueryData.argtypes = [c.c_void_p]
            self.p.PdhGetFormattedCounterArrayW.argtypes = [c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.POINTER(w.DWORD), c.c_void_p]
            self.p.PdhCloseQuery.argtypes = [c.c_void_p]
            if self.p.PdhOpenQueryW(None, 0, c.byref(self.query)) or self.p.PdhAddEnglishCounterW(self.query, r'\Thermal Zone Information(*)\Temperature', 0, c.byref(self.counter)):
                self.close();return
            self.p.PdhCollectQueryData(self.query)
        except (OSError, AttributeError):self.close()

    def read(self):
        if not self.query:return thermal_sample([])
        class Value(c.Structure):_fields_ = [('status', w.DWORD), ('value', c.c_double)]
        class Item(c.Structure):_fields_ = [('name', w.LPWSTR), ('data', Value)]
        if self.p.PdhCollectQueryData(self.query):return thermal_sample([])
        size, count = w.DWORD(), w.DWORD()
        self.p.PdhGetFormattedCounterArrayW(self.counter, 0x200, c.byref(size), c.byref(count), None)
        if not size.value or size.value > 4 * 1024 * 1024:return thermal_sample([])
        buf = c.create_string_buffer(size.value)
        if self.p.PdhGetFormattedCounterArrayW(self.counter, 0x200, c.byref(size), c.byref(count), buf):return thermal_sample([])
        items = c.cast(buf, c.POINTER(Item))
        return thermal_sample([(items[i].name or '', items[i].data.value, items[i].data.status) for i in range(count.value)])

    def close(self):
        if self.query:self.p.PdhCloseQuery(self.query);self.query = None
