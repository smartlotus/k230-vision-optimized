"""Three small native-decoder transfers, readback and cleanup."""
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/temporal_camera'))
from native_smoke import CanMvClient, execute_marked, runtime_append_script

c = CanMvClient()
nonce = uuid.uuid4().hex
root = '/tmp/k230_temporal_v1_tprobe3'
marker = 'TP_MK_' + nonce + ':'
execute_marked(c, "import os,json\nos.mkdir(" + repr(root) + ")\n"
               "print(" + repr(marker) + "+json.dumps({'ok':True}))", marker)
block = bytes(range(256)) * 32
results = []
for index in range(3):
    marker = 'TP_CHUNK_' + nonce + '_' + str(index) + ':'
    result, _ = execute_marked(c, runtime_append_script(
        root + '/p.bin', index * len(block), block, marker), marker)
    results.append(result)
script = """import os,json,hashlib,ubinascii
p=PROBE_FILE
with open(p,'rb') as f:b=f.read()
r={'bytes':len(b),'sha256':ubinascii.hexlify(hashlib.sha256(b).digest()).decode()}
os.remove(p)
os.rmdir(PROBE_ROOT)
print(FINAL_MARKER+json.dumps(r))
"""
marker = 'TP_FINAL_' + nonce + ':'
script = script.replace('PROBE_FILE', repr(root + '/p.bin')).replace('PROBE_ROOT', repr(root)).replace('FINAL_MARKER', repr(marker))
result, _ = execute_marked(c, script, marker)
assert result == {'bytes': 24576, 'sha256': hashlib.sha256(block * 3).hexdigest()}
receipt = dict(status='PASS', blocks=results, final=result, block_bytes=8192,
               decoder='ubinascii', cleanup='own probe removed')
(Path(__file__).parent / 'SMALL_TRANSFER_PROBE.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
