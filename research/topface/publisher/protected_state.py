"""Current-user DPAPI storage for transfer recovery, never committed.

Resumable-session URLs and signed source URLs are sensitive. Public receipts
store identities only; protected recovery state retains the original session.
"""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path


class Blob(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]


def _blob(data):
    buffer=ctypes.create_string_buffer(data,len(data))
    return Blob(len(data),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte))),buffer


def transform(data,entropy,decrypt=False):
    if os.name!='nt':raise RuntimeError('Host recovery requires Windows current-user DPAPI')
    incoming,inbuffer=_blob(data);salt,saltbuffer=_blob(entropy);outgoing=Blob()
    crypt=ctypes.WinDLL('crypt32',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    if decrypt:
        function=crypt.CryptUnprotectData
        function.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
        success=function(ctypes.byref(incoming),None,ctypes.byref(salt),None,None,1,ctypes.byref(outgoing))
    else:
        function=crypt.CryptProtectData
        function.argtypes=[ctypes.POINTER(Blob),wintypes.LPCWSTR,ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
        success=function(ctypes.byref(incoming),'K230 transfer recovery',ctypes.byref(salt),None,None,1,ctypes.byref(outgoing))
    function.restype=wintypes.BOOL
    if not success:raise RuntimeError('DPAPI current-user protection failed')
    try:return ctypes.string_at(outgoing.data,outgoing.size)
    finally:
        kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
        kernel.LocalFree(ctypes.cast(outgoing.data,ctypes.c_void_p))


def save(path,value,run_id):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encrypted=transform(value,run_id.encode('ascii'))
    temporary=path.with_suffix(path.suffix+'.part');temporary.write_bytes(encrypted);temporary.replace(path)


def load(path,run_id):return transform(Path(path).read_bytes(),run_id.encode('ascii'),decrypt=True)
