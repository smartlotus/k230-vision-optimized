"""Append-only SD generations; incomplete or corrupt newest files never win."""
import os,json,hashlib,binascii
from .schema import validate,clone


def sha(raw):return binascii.hexlify(hashlib.sha256(raw).digest()).decode()


class ProfileStore:
    def __init__(self,root,limit=128):
        self.root=root.rstrip('/');self.limit=limit;self.warnings=[]

    def _names(self,slot):
        if type(slot) is not int or not 1<=slot<=6:raise ValueError('Profile slot1..6 required')
        try:names=os.listdir(self.root)
        except OSError:return []
        prefix='slot%d-' % slot
        return sorted(n for n in names if n.startswith(prefix) and n.endswith('.json') and len(n)==19 and n[6:14].isdigit())

    def _prepare(self):
        current=''
        for part in self.root.split('/'):
            if not part:continue
            current+=('/' if self.root.startswith('/') or current else '')+part
            try:os.mkdir(current)
            except OSError:
                if not os.stat(current)[0]&0x4000:raise

    def _commit(self,name,data):
        self._prepare();temporary=name+'.tmp'
        with open(self.root+'/'+temporary,'w') as stream:
            stream.write(data)
            try:stream.flush()
            except AttributeError:pass
        try:os.sync()
        except AttributeError:pass
        with open(self.root+'/'+temporary,'r') as stream:readback=stream.read(131073)
        if readback!=data:raise OSError('Profile readback differs')
        os.rename(self.root+'/'+temporary,self.root+'/'+name)
        try:os.sync()
        except AttributeError:pass

    def _selections(self):
        try:names=os.listdir(self.root)
        except OSError:return []
        return sorted(n for n in names if len(n)==22 and n.startswith('selected-') and n.endswith('.json') and n[9:17].isdigit())

    def selected_slot(self):
        for name in reversed(self._selections()):
            try:
                with open(self.root+'/'+name) as stream:record=json.loads(stream.read(2049))
                if (set(record)=={'generation','slot','sha256'} and type(record['slot']) is int and 1<=record['slot']<=6
                        and record['generation']==int(name[9:17]) and record['sha256']==sha(str(record['slot']).encode())):
                    return record['slot']
            except (OSError,ValueError,TypeError,KeyError):pass
        return 1

    def select_slot(self,slot):
        self._names(slot)
        if self.selected_slot()==slot:return
        names=self._selections()
        if len(names)>=self.limit:raise OSError('Selection history full')
        generation=max([int(n[9:17]) for n in names] or [0])+1
        name='selected-%08d.json' % generation
        self._commit(name,json.dumps({'generation':generation,'slot':slot,'sha256':sha(str(slot).encode())}))
        if self.selected_slot()!=slot:raise OSError('Selected slot readback differs')

    def _read(self,name):
        with open(self.root+'/'+name,'r') as stream:raw=stream.read(131073)
        if len(raw)>131072:raise ValueError('Profile too large')
        envelope=json.loads(raw)
        if (not isinstance(envelope,dict) or set(envelope)!={'schema_version','generation','sha256','payload'}
                or envelope['schema_version']!=1 or not isinstance(envelope['payload'],str)
                or not isinstance(envelope['sha256'],str)):raise ValueError('Envelope schema mismatch')
        if sha(envelope['payload'].encode())!=envelope['sha256']:raise ValueError('Profile checksum mismatch')
        if type(envelope['generation']) is not int or envelope['generation']!=int(name[6:14]):raise ValueError('Generation mismatch')
        profile=json.loads(envelope['payload']);validate(profile)
        return {'profile':profile,'generation':envelope['generation'],'sha256':envelope['sha256'],'file':name}

    def load(self,slot=1,previous=False):
        self.warnings=[];valid=[]
        for name in reversed(self._names(slot)):
            try:valid.append(self._read(name))
            except (OSError,ValueError,TypeError,KeyError) as error:self.warnings.append(name+': '+str(error))
            if len(valid)>= (2 if previous else 1):break
        index=1 if previous else 0
        return valid[index] if len(valid)>index else None

    def save(self,profile,slot=1):
        validate(profile);names=self._names(slot)
        if len(names)>=self.limit:raise OSError('History full; export/archive before further saves')
        generation=max([int(n[6:14]) for n in names] or [0])+1
        if generation>99999999:raise ValueError('Generation exhausted')
        payload=json.dumps(profile);digest=sha(payload.encode())
        final='slot%d-%08d.json' % (slot,generation)
        data=json.dumps({'schema_version':1,'generation':generation,'sha256':digest,'payload':payload})
        self._commit(final,data)
        result=self._read(final)
        if result['sha256']!=digest:raise OSError('Committed profile readback differs')
        return result
