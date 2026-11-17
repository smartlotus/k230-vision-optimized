import os,json,gc,hashlib,ubinascii
r={'nonce':'TEMPORAL_NATIVE_FINAL_20261117','files':{},'other_tmp_files':{},'own_probe_cleanup':[]}
jobs={"/tmp/k230_temporal_v1_c1r3/camera_xy.kmodel":{"bytes":2646600,"sha256":"ac86c72a648df1c9f484c920180b0bcb9ae877f26d20c2cea364969d0d921fbc"},"/tmp/k230_temporal_v1_c1r3/KMODEL_REPORT.json":{"bytes":12269,"sha256":"faa72c66564181372e245dd6e4d051ebad80f1637457aae3e98b6997b28561a9"},"/tmp/k230_temporal_v1_c1r3/board_helpers.py":{"bytes":2556,"sha256":"0fdb8b81c7a988bd8d39bc1f5fbbfb8aa7699905f6c85fe5a1aaeaf0fe01fd9f"},"/tmp/k230_temporal_v1_c1r3/board_multiroute.py":{"bytes":10385,"sha256":"755c424a9ed52ed5b4ffe8eefa34be330f343193ddc2e30317a676dc3ca76f0e"},"/tmp/k230_temporal_v1_c1r3/board_temporal.py":{"bytes":4187,"sha256":"7b2245f4712fd13a3f8ea39861a2d896e87e8ead1da704da1f4434046dad7cd9"},"/tmp/k230_temporal_v1_c1r3/competition_paths.py":{"bytes":4575,"sha256":"cb42cd555d452c6781caa43d5ae9084c0dadbe14d6741aa985edca42c8eb5a06"},"/tmp/k230_temporal_v1_c1r3/embedded_filter.py":{"bytes":7656,"sha256":"856676f28fc032e52bf4916850b2c3819f05bbe74ea1be59d2844f3b16b77402"},"/tmp/k230_temporal_v1_c1r3/localization_runtime.py":{"bytes":3420,"sha256":"196c2e2c7518f986ca779ade2a9e9be42d6130a801705875c6b47cdbecd96bbe"},"/tmp/k230_temporal_v1_c1r3/deployment.json":{"bytes":100016,"sha256":"641eab171629e6b44f43f6222409ca093801132af692a436810d16258fd26326"}}
others=[{"path":"/tmp/xyusb1117d/capture30.json","bytes":285,"sha256":"d8e9f1a3127157a2ecf7ed7da638fe5091602cf84836be7575e05dd587e58e49"},{"path":"/tmp/xyusb1117d/write-read-test.bin","bytes":8192,"sha256":"dc404a613fedaeb54034514bc6505f56b933caa5250299ba7d094377a51caa46"},{"path":"/tmp/xyusb1117f/postflight.json","bytes":525,"sha256":"1743eefdbe7a717c4fb59fbb7d69468967d90b6a2543630b94189dc9b285295c"},{"path":"/tmp/xyusb1117f/restart30-f03.json","bytes":4022,"sha256":"d32d47abe902b331a595bab69c7c74ba4b7b7f7ab763c8d0b5ecae6248050b45"},{"path":"/tmp/xyusb1117f/chain300-f02.json","bytes":6688,"sha256":"0504243e3651c93c01583a634630b1e2a733d945ebfd65f30bd3d3acaacad2b9"},{"path":"/tmp/xyusb1117f/chain30-f01.json","bytes":4025,"sha256":"7a28e000df73f522ba23c420432a58f5249f4c3260827d45eca2e93edbe19128"},{"path":"/tmp/xyusb1117f/board_usb_smoke.py","bytes":13774,"sha256":"476c0eba7d4fd42e8d81ae01a3fec4db335dd530df81a7d16592d477062625e6"},{"path":"/tmp/xyusb1117f/engineering.kmodel","bytes":2133296,"sha256":"ae6f472c020f22026c4e6779c5899a38296401465731a258be557061d391a40b"},{"path":"/tmp/xyusb1117f/engineering.json","bytes":52052,"sha256":"3d923840a98907c3569678b7354fd102a166162bf4a0f94e36fb3047de359792"},{"path":"/tmp/xyusb1117f/board_probe.py","bytes":11315,"sha256":"51aaa38e69bcba06a561236bcd4ce4d8e8479e42c9ba180348218f0c5ed710b4"}]
def check(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while True:
   b=f.read(8192)
   if not b:break
   h.update(b)
 return {'bytes':os.stat(p)[6],'sha256':ubinascii.hexlify(h.digest()).decode()}
for p,expected in jobs.items():
 got=check(p);got['matches']=got['bytes']==expected['bytes'] and got['sha256']==expected['sha256'];r['files'][p]=got
for row in others:
 got=check(row['path']);got['matches']=got['bytes']==row['bytes'] and got['sha256']==row['sha256'];r['other_tmp_files'][row['path']]=got
r['startup']=check('/sdcard/main.py')
r['boot_present']='boot.py' in os.listdir('/sdcard')
r['approved_old_tmp_removed']='xyusb1117e' not in os.listdir('/tmp')
for p in ['/tmp/k230_temporal_v1_c1r1117','/tmp/k230_temporal_v1_c1r2','/tmp/k230_temporal_v1_tprobe2']:
 try:
  if not os.listdir(p):os.rmdir(p);r['own_probe_cleanup'].append(p)
 except OSError:pass
p='/tmp/k230_temporal_v1_tprobe'
try:
 if os.listdir(p)==['p.bin']:
  got=check(p+'/p.bin')
  expected=ubinascii.hexlify(hashlib.sha256(bytes(range(256))*64).digest()).decode()
  if got['bytes']==16384 and got['sha256']==expected:
   os.remove(p+'/p.bin');os.rmdir(p);r['own_probe_cleanup'].append(p)
except OSError:pass
gc.collect()
r['final_mmz']=list(gc.sys_mmz())
r['python_heap_free']=gc.mem_free()
r['runtime_tmp_entries']=os.listdir('/tmp')
r['sdcard_entries']=os.listdir('/sdcard')
r['status']='PASS' if all(v['matches'] for v in r['files'].values()) and all(v['matches'] for v in r['other_tmp_files'].values()) else 'MISMATCH'
with open('/tmp/k230_temporal_v1_c1r3/postflight.json','w') as f:f.write(json.dumps(r))
print('TEMPORAL_NATIVE_FINAL:'+json.dumps(r))
