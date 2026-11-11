"""Run stdlib tests and persist actual output and input source fingerprints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def run_tests(output):
    command=[sys.executable,'-B','-X','utf8','-m','unittest','discover','-s','tests','-v']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    log=output.with_suffix('.log')
    log.write_text(result.stdout+result.stderr,encoding='utf-8')
    sources={}
    for name in ('run_board.py','run_multicam_preview.py','run_block_preview.py','run_block_service.py','run_vision_preview.py'):
        sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for directory in ('core','vision','ports','k230','sim','tests','tools','stm32'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.suffix in ('.py','.c','.h') and '__pycache__' not in path.parts:
                sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':command,
            'exit_code':result.returncode,'scope':'Host offline regression only; not K230/STM32 hardware acceptance',
            'stdout_log':log.name,'source_sha256':sources}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(result.stdout+result.stderr)
    return result.returncode


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'evidence'/'validation.json')
    raise SystemExit(run_tests(parser.parse_args().output))
