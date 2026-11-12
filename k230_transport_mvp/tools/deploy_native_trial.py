"""Host-only isolated deployment over the configured official CanMV MCP."""
import argparse
import base64
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import tomllib
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[2]


class CanMvClient:
    def __init__(self):
        config=tomllib.loads((Path(os.getenv('CODEX_HOME') or Path.home()/'.codex')/'config.toml').read_text(encoding='utf-8'))
        entry=config['mcp_servers']['canmv-k230']
        if entry.get('enabled') is False:raise ValueError('MCP disabled')
        self.url=entry['url'];url=urllib.parse.urlsplit(self.url)
        local=url.hostname=='localhost' or ipaddress.ip_address(url.hostname).is_loopback
        if not local or url.scheme!='http' or url.path!='/mcp' or url.username or url.password:
            raise ValueError('Official loopback MCP required')
        self.headers=dict(entry.get('http_headers',{}))
        for key,name in entry.get('env_http_headers',{}).items():
            if os.getenv(name):self.headers[key]=os.environ[name]
        if entry.get('bearer_token_env_var'):
            self.headers['Authorization']='Bearer '+os.environ[entry['bearer_token_env_var']]
        if not any(k.lower()=='authorization' and v for k,v in self.headers.items()):raise ValueError('MCP auth missing')
        self.headers.update({'Content-Type':'application/json','Accept':'application/json, text/event-stream'})
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));self.sequence=0
        self.request('initialize',{'protocolVersion':'2024-11-05','capabilities':{},
            'clientInfo':{'name':'k230-isolated-native-trial','version':'1'}})

    def request(self,method,params):
        self.sequence+=1
        data=json.dumps({'jsonrpc':'2.0','id':self.sequence,'method':method,'params':params}).encode()
        request=urllib.request.Request(self.url,data=data,headers=self.headers)
        with self.opener.open(request,timeout=30) as response:value=json.load(response)
        if value.get('id')!=self.sequence or 'error' in value:raise RuntimeError('MCP outcome UNKNOWN; no retry')
        return value['result']

    def tool(self,name,args=None):
        value=self.request('tools/call',{'name':name,'arguments':args or {}})
        if value.get('isError'):raise RuntimeError(name+' failed; no retry')
        return json.loads(value['content'][0]['text'])


def deploy(remote,output):
    if not remote.startswith(('/sdcard/transport_mvp_optimized_','/sdcard/kopt_')) or '..' in remote or '/' in remote[len('/sdcard/'):]:
        raise ValueError('Explicit isolated test directory required')
    client=CanMvClient()
    if client.tool('canmv_script_running').get('running') is not False:raise RuntimeError('Board busy; no writes')
    if client.tool('canmv_stat_file',{'path':remote}).get('exists') is not False:
        raise RuntimeError('New directory required; existing install preserved')
    files=[]
    for folder in ('core','vision','ports','k230'):
        files.extend(p for p in (ROOT/'k230_transport_mvp'/folder).rglob('*.py') if '__pycache__' not in p.parts)
    files.extend(ROOT/'k230_transport_mvp'/name for name in ('tools/__init__.py','tools/native_fps_trial.py'))
    items=[(p.relative_to(ROOT/'k230_transport_mvp').as_posix(),p.read_bytes()) for p in sorted(files)]
    items.append(('protected-state.json',(ROOT/'docs/validation/native-fps-20261112/protected-state.json').read_bytes()))
    if any(len((remote+'/'+name).encode('utf-8'))>=64 for name,_ in items):
        raise ValueError('Use a shorter isolated root: native protocol paths must stay below64 bytes')
    directories={remote}
    for name,_ in items:
        parent=Path(name).parent
        while str(parent)!='.':directories.add(remote+'/'+parent.as_posix());parent=parent.parent
    for path in sorted(directories,key=lambda v:(v.count('/'),v)):
        result=client.tool('canmv_mkdir',{'path':path})
        if result.get('success') is not True:raise RuntimeError('Directory creation not confirmed')
    output.parent.mkdir(parents=True,exist_ok=True)
    receipt={'remote_root':remote,'files':[],'status':'INCOMPLETE','board_running_before':False}
    for index,(name,raw) in enumerate(items):
        path=remote+'/'+name
        result=client.tool('canmv_write_file',{'path':path,'encoding':'base64','content':base64.b64encode(raw).decode()})
        if result.get('success') is not True:raise RuntimeError('Write not confirmed; no retry')
        read=client.tool('canmv_read_file',{'path':path,'encoding':'base64'})
        actual=base64.b64decode(read['content'],validate=True)
        row={'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'readback_sha256':hashlib.sha256(actual).hexdigest()}
        receipt['files'].append(row)
        output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        if raw!=actual:raise RuntimeError('Native bytes mismatch')
        if (index+1)%10==0:print(json.dumps({'verified_files':index+1,'total':len(items)}),flush=True)
    receipt['status']='VERIFIED_BYTES_ONLY'
    output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'files':len(items),'root':remote}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();deploy(args.remote,args.output)
