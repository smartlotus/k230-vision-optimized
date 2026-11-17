"""Read every member of the shipped archives and verify the external SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(stream):
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path):
    receipt = json.loads(path.with_suffix('.SHA256.json').read_text(encoding='utf8'))
    with path.open('rb') as stream:
        if path.stat().st_size != receipt['bytes'] or sha(stream) != receipt['sha256']:
            raise ValueError('Archive identity differs for ' + path.name)
    with zipfile.ZipFile(path) as archive:
        inventory = json.loads(archive.read('SHA256_MANIFEST.json'))
        names = archive.namelist()
        if (len(names) != len(set(names)) or len(inventory) != receipt['verified_members']
                or set(names) != set(inventory) | {'SHA256_MANIFEST.json'}):
            raise ValueError('Archive member inventory differs for ' + path.name)
        for name, entry in inventory.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe archive member ' + name)
            with archive.open(name) as stream:
                actual = sha(stream)
            if archive.getinfo(name).file_size != entry['bytes'] or actual != entry['sha256']:
                raise ValueError('File integrity differs for ' + name)
    return {'archive': path.name, 'status': 'ALL_MEMBERS_VERIFIED', 'members': len(inventory), 'sha256': receipt['sha256']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archives', type=Path, nargs='+')
    args = parser.parse_args()
    for path in args.archives:
        print(json.dumps(verify(path)))


if __name__ == '__main__':
    main()
