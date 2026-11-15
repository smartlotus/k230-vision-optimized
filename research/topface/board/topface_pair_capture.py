"""Bounded MicroPython same-frame tensor evidence; no performance claims."""
import json
import os
import struct


def bytes_record(path,shape,dtype,layout):
    from topface_trial import digest
    return {'name':path.rsplit('/',1)[-1],'bytes':os.stat(path)[6],'sha256':digest(path),
            'shape':list(shape),'dtype':dtype,'layout':layout}


def write_planar(path,array,height,width):
    shape=tuple(array.shape)
    if shape not in ((3,height,width),(1,3,height,width)):
        raise ValueError('RGB planar tensor shape differs from the frozen file contract')
    # Explicit indexing avoids assuming ULAb exposes a contiguous buffer API.
    data=bytearray(3*height*width);offset=0
    for channel in range(3):
        for y in range(height):
            for x in range(width):
                value=array[channel,y,x] if len(shape)==3 else array[0,channel,y,x]
                integer=int(value)
                if integer!=value or not 0<=integer<=255:raise ValueError('RGB input is not uint8-valued')
                data[offset]=integer;offset+=1
    with open(path,'wb') as stream:stream.write(data)
    return bytes_record(path,[1,3,height,width],'uint8','NCHW_RGB')


def write_raw(path,array,positions):
    if tuple(array.shape)!=(1,positions,50):raise ValueError('Unexpected raw KPU tensor shape')
    data=bytearray(positions*50*4);offset=0
    for point in range(positions):
        for channel in range(50):
            struct.pack_into('<f',data,offset,float(array[0,point,channel]));offset+=4
    with open(path,'wb') as stream:stream.write(data)
    return bytes_record(path,[1,positions,50],'float32_le','C')


class PairCapture:
    def __init__(self,root,identity,runtime,expected):
        from topface_trial import digest
        if (type(expected) is not int or not 1<=expected<=3 or identity['model_wh']!=[320,192] or
                identity['output_shape']!=[1,1275,50] or identity['input']!='NCHW_RGB_UINT8'):
            raise ValueError('Pair capture requires the bounded formal320x192 contract')
        for suffix in ('PAIR_CAPTURE_RESULT.json','pair-01','pair-02','pair-03'):
            try:os.stat(root+'/'+suffix)
            except OSError:pass
            else:raise ValueError('Preserve existing pair evidence; use a fresh independent trial directory')
        self.root=root;self.identity=identity;self.runtime=str(runtime);self.expected=expected;self.records=[]
        self.contract_sha256=digest(root+'/output_contract.json')

    def start(self,source,prepared,sequence,capture_ticks_us):
        number=len(self.records)+1;directory=self.root+'/pair-%02d'%number
        try:os.stat(directory)
        except OSError:os.mkdir(directory)
        else:raise ValueError('Preserve existing pair files; use a new independent trial directory')
        record={'status':'INPUTS_SAVED_WAITING_KPU_OUTPUT','sample':number,
            'camera_id':2,'capture_sequence_index':sequence,'capture_ticks_us':capture_ticks_us,
            'hardware_frame_id':'UNAVAILABLE','runtime':self.runtime,
            'model_sha256':self.identity['kmodel_sha256'],'reference_onnx_sha256':self.identity['onnx_sha256'],
            'output_contract_sha256':self.contract_sha256,'input_frozen_before_AI2D':True,
            'source_observed_shape':list(source.shape),'performance_measurement':False,
            'source_observed_dtype':str(source.dtype),'prepared_observed_dtype':str(prepared.dtype),
            'files':{'source':write_planar(directory+'/source.rgb888p',source,360,640),
                     'prepared':write_planar(directory+'/prepared.rgb888p',prepared,192,320)}}
        with open(directory+'/sample.json','w') as stream:stream.write(json.dumps(record))
        return directory,record

    def finish(self,pending,raw,decoded):
        directory,record=pending
        record['files']['raw']=write_raw(directory+'/kpu-raw.f32le',raw,1275)
        record['raw_observed_dtype']=str(raw.dtype)
        with open(directory+'/decoded.json','w') as stream:stream.write(json.dumps(decoded))
        from topface_trial import digest
        record['files']['decoded']={'name':'decoded.json','bytes':os.stat(directory+'/decoded.json')[6],
                                     'sha256':digest(directory+'/decoded.json')}
        record['status']='SAME_FRAME_INPUT_AND_KPU_RAW_SAVED'
        with open(directory+'/sample.json','w') as stream:stream.write(json.dumps(record))
        self.records.append(record)

    def result(self):
        try:os.stat(self.root+'/PAIR_CAPTURE_RESULT.json')
        except OSError:pass
        else:raise ValueError('Preserve the original pair capture index')
        result={'status':'PAIRED_CAPTURE_COMPLETE' if len(self.records)==self.expected else 'PAIRED_CAPTURE_INCOMPLETE',
            'requested':self.expected,'completed':len(self.records),'samples':self.records,
            'performance_measurement':False,'host_numerical_comparison':'NOT_RUN'}
        with open(self.root+'/PAIR_CAPTURE_RESULT.json','w') as stream:stream.write(json.dumps(result))
        return result
