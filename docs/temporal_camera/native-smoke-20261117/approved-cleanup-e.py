import os,json,hashlib,binascii
root='/tmp/xyusb1117e'
files=[{"path":"/tmp/xyusb1117e/format-probe.json","bytes":138,"sha256":"9bd53dd29fbe68c88d7e66074ba6e93f164d019c17b1e8fff3fc619ad095328b"},{"path":"/tmp/xyusb1117e/format-probe-rgb888.bin","bytes":691200,"sha256":"05ff7883dbda98c11af87bdc2ea440b3ad218a5fdb0c121be873a1f6bffd7bc9"},{"path":"/tmp/xyusb1117e/format-probe-raw.jpg","bytes":85162,"sha256":"ad01a03343e5bf64d307e026ef9ae305cad322633bd9079caa3270cb4bd674df"},{"path":"/tmp/xyusb1117e/chain300-e02.json","bytes":6649,"sha256":"72421013aa5e2fedc26e17b187ccab06af64148e59d95fa41d4f2ea779725202"},{"path":"/tmp/xyusb1117e/chain30-e01.json","bytes":3993,"sha256":"2200a69e9ec4a7a484161046a28fca68bfa62045390b01be445b60a22798d0a2"},{"path":"/tmp/xyusb1117e/engineering.kmodel","bytes":2133296,"sha256":"ae6f472c020f22026c4e6779c5899a38296401465731a258be557061d391a40b"},{"path":"/tmp/xyusb1117e/board_usb_smoke.py","bytes":13205,"sha256":"852bc8576bd5e1c40737eb8decab918a032910b30d6ea5e047f2f110392dc357"},{"path":"/tmp/xyusb1117e/engineering.json","bytes":52052,"sha256":"3d923840a98907c3569678b7354fd102a166162bf4a0f94e36fb3047de359792"},{"path":"/tmp/xyusb1117e/board_probe.py","bytes":11315,"sha256":"51aaa38e69bcba06a561236bcd4ce4d8e8479e42c9ba180348218f0c5ed710b4"}]
if set(os.listdir(root))!=set(row['path'].split('/')[-1] for row in files):raise ValueError('Directory contents changed; no deletion')
for row in files:
 p=row['path']
 if not p.startswith(root+'/') or p.count('/')!=3:raise ValueError('Cleanup path outside approved directory')
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while True:
   b=f.read(8192)
   if not b:break
   h.update(b)
 if os.stat(p)[6]!=row['bytes'] or binascii.hexlify(h.digest()).decode()!=row['sha256']:raise ValueError('Stored file changed; no deletion')
deleted=[]
for row in files:
 os.remove(row['path'])
 deleted.append(row['path'])
os.rmdir(root)
r={'deleted':deleted,'approved_directory':root,'remaining_tmp':os.listdir('/tmp'),'statvfs':list(os.statvfs('/tmp'))}
print('TEMPORAL_CLEANUP:'+json.dumps(r))
