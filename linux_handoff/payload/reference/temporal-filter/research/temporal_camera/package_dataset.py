"""Archive the delivered RGB and class masks with a verified member inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from dataset import digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--map',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as bundle:
        members={}
        for prefix,directory in (('smoke',args.smoke),('pilot',args.pilot),('map',args.map)):
            for path in sorted(directory.rglob('*')):
                if not path.is_file() or path.name.endswith('.incomplete.jsonl'):
                    continue
                if path.is_symlink():
                    raise ValueError('Symlink in delivery inputs')
                member=prefix+'/'+path.relative_to(directory).as_posix()
                members[member]={'sha256':digest(path),'bytes':path.stat().st_size}
                bundle.write(path,member)
        bundle.writestr('SHA256_MANIFEST.json',json.dumps(members,indent=2)+'\n')
    with zipfile.ZipFile(args.out) as bundle:
        if set(bundle.namelist())!=set(members)|{'SHA256_MANIFEST.json'}:
            raise ValueError('Delivery inventory differs')
        for member,expected in members.items():
            with bundle.open(member) as stream:
                actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected['sha256'] or bundle.getinfo(member).file_size!=expected['bytes']:
                raise ValueError('Archive file digest differs: '+member)
    receipt={'status':'ALL_MEMBERS_VERIFIED','archive':args.out.name,'sha256':digest(args.out),
             'bytes':args.out.stat().st_size,'verified_members':len(members)}
    args.out.with_suffix('.SHA256.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
