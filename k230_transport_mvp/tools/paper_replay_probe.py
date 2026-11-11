"""Bounded native replay of one saved JPEG; no live/physical acceptance."""


def probe(path='/sdcard/transport_mvp/ellipse_perf_samples/sample-000014-live.jpg'):
    import image,time,hashlib,binascii,gc
    from vision.ring_native import detect_paper_targets,detect_paper_circle_proposals
    from vision.ellipse_native import configure_boundary_backend
    configure_boundary_backend(True)
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        while True:
            chunk=stream.read(16384)
            if not chunk:break
            digest.update(chunk)
    frame=image.Image(path)
    if (frame.width(),frame.height())!=(640,360):raise ValueError('Expected640x360 saved frame')
    roi=(210,4,420,310);region=frame.copy(roi=roi).to_grayscale();gray=region.to_numpy_ref()
    rows=[]
    for name,detector in (('FULL_CONTOUR',detect_paper_targets),('EXPERIMENTAL_CIRCLE_PROPOSAL',detect_paper_circle_proposals)):
        timings=[]
        for _ in range(3):
            start=time.ticks_ms();result=detector(gray,roi,pixel_scale=2)
            timings.append(time.ticks_diff(time.ticks_ms(),start))
        rows.append({'detector':name,'timings_ms':timings,'groups':result['groups'],
                     'reason':result['reason'],'complete':result['complete'],
                     'raw_circle_count':result.get('raw_circle_count'),
                     'proposal_only':name!='FULL_CONTOUR'})
    frame=region=gray=None;gc.collect()
    return {'scope':'NATIVE_REPLAY_OF_SAVED_JPEG','path':path,
            'jpeg_sha256':binascii.hexlify(digest.digest()).decode(),'rows':rows,
            'input_source':'REAL_CAMERA_SAVED_JPEG_NOT_NEW_LIVE_FRAMES',
            'physical_action':False,'complete_paper_recognition_verified':False,
            'experimental_proposals_used_for_alignment':False}


if __name__=='__main__':
    import sys,json
    if '/sdcard/transport_mvp' not in sys.path:sys.path.insert(0,'/sdcard/transport_mvp')
    try:result=probe()
    except Exception as error:result={'scope':'NATIVE_REPLAY_OF_SAVED_JPEG','status':'FAILED','error':str(error)}
    print('PAPER_REPLAY_PROBE:'+json.dumps(result))
