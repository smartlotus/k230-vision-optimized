import sys,json,os
root='/data/k230_temporal_v1_sd1118'
sys.path.insert(0,root)
entry={'status':'STARTED','root':root,'run_id':'sd1118_10b'}
try:
 from board_multiroute import run
 result=run(root,'sd1118_10b','openaicam#openaicam',frames=10,mode='filtered')
 entry['status']='RETURNED'
 entry['report_status']=result['status']
except BaseException as error:
 entry['status']='ENTRY_FAILED'
 entry['error']=str(error)
 sys.print_exception(error)
finally:
 with open('/data/k230_temporal_v1_sd1118/sd1118_10b-entry.json','w') as stream:stream.write(json.dumps(entry))
 print('TEMPORAL_ENTRY:'+json.dumps(entry))
