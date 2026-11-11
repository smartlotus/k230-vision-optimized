"""Verify imported bytes and run the host suite from this independent repo."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/validation')
    parser.add_argument('--sources-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT/'docs/UPSTREAM_IMPORT.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    mismatches = []
    for row in manifest['copied_files']:
        path = (ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != row['sha256']:
            mismatches.append(row['path'])
    report = {'source_commit': manifest['source_commit'], 'manifest_sha256': digest(manifest_path),
              'files_checked': len(manifest['copied_files']), 'mismatches': mismatches,
              'scope': 'Imported byte identities only; not hardware acceptance'}
    (args.output/'imported-sources.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    if mismatches:
        print(json.dumps(report))
        return 1
    if args.sources_only:
        print(json.dumps(report))
        return 0
    output = (args.output/'host-tests.json').resolve()
    command = [sys.executable, '-B', '-X', 'utf8', '-m', 'tools.export_test_report', '--output', str(output)]
    result = subprocess.run(command, cwd=ROOT/'k230_transport_mvp', capture_output=True, text=True, encoding='utf-8')
    log = output.with_suffix('.log').read_text(encoding='utf-8') if output.with_suffix('.log').exists() else result.stdout+result.stderr
    count = re.search(r'Ran (\d+) tests', log)
    summary = {'exit_code': result.returncode, 'tests_run': int(count[1]) if count else None,
               'skipped': len(re.findall(r'\.\.\. skipped ', log)),
               'python': sys.version, 'repository_root': str(ROOT), 'board_operations': 0,
               'native_25fps_verified': False}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary))
    if result.returncode:
        print(log[-10000:])
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
