import sys,json,os
root='/data/k230_temporal_v1_cmp1118_tcn4'
sys.path.insert(0,root)
entry={'status':'STARTED','root':root,'run_id':'tcn4_60b'}
try:
 from board_multiroute import run
 result=run(root,'tcn4_60b','openaicam#openaicam',frames=60,mode='filtered')
 entry['status']='RETURNED'
 entry['report_status']=result['status']
except BaseException as error:
 entry['status']='ENTRY_FAILED'
 entry['error']=str(error)
 sys.print_exception(error)
finally:
 with open('/data/k230_temporal_v1_cmp1118_tcn4/tcn4_60b-entry.json','w') as stream:stream.write(json.dumps(entry))
 print('TEMPORAL_ENTRY:'+json.dumps(entry))
