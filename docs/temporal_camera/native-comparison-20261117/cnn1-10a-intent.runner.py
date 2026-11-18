import sys,json,os
root='/data/k230_temporal_v1_cmp1118_cnn1'
sys.path.insert(0,root)
entry={'status':'STARTED','root':root,'run_id':'cnn1_10a'}
try:
 from board_multiroute import run
 result=run(root,'cnn1_10a','openaicam#openaicam',frames=10,mode='filtered')
 entry['status']='RETURNED'
 entry['report_status']=result['status']
except BaseException as error:
 entry['status']='ENTRY_FAILED'
 entry['error']=str(error)
 sys.print_exception(error)
finally:
 with open('/data/k230_temporal_v1_cmp1118_cnn1/cnn1_10a-entry.json','w') as stream:stream.write(json.dumps(entry))
 print('TEMPORAL_ENTRY:'+json.dumps(entry))
