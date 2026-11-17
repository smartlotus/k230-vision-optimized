import sys,json,os
root='/tmp/k230_temporal_v1_c1r3'
sys.path.insert(0,root)
entry={'status':'STARTED','root':root,'run_id':'c1_10e'}
try:
 from board_multiroute import run
 result=run(root,'c1_10e','openaicam#openaicam',frames=10,mode='filtered')
 entry['status']='RETURNED'
 entry['report_status']=result['status']
except BaseException as error:
 entry['status']='ENTRY_FAILED'
 entry['error']=str(error)
 sys.print_exception(error)
finally:
 with open('/tmp/k230_temporal_v1_c1r3/c1_10e-entry.json','w') as stream:stream.write(json.dumps(entry))
 print('TEMPORAL_ENTRY:'+json.dumps(entry))
