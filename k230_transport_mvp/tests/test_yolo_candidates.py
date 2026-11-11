import unittest
import numpy as np
from vision.yolo_candidates import input_contract,decode_two_class_proposals


class YoloCandidateContractTests(unittest.TestCase):
    def output(self,size=320):return np.zeros((1,6,input_contract(size)['positions']),np.float32)

    def test_fixed_resize_uses_integer_shapes_and_declared_padding(self):
        for size,height,pad,n in ((224,126,49,1029),(320,180,70,2100)):
            c=input_contract(size)
            self.assertEqual(c['resized_wh'],[size,height]);self.assertEqual(c['pad_tblr'],[pad,pad,0,0])
            self.assertEqual(c['positions'],n);self.assertEqual(c['pad_rgb'],[114]*3)
            self.assertEqual(input_contract(size,'top_left',128)['pad_tblr'],[0,pad*2,0,0])

    def test_overlapping_classes_keep_full_native_rows_and_source_coordinates(self):
        for size in (224,320):
            c=input_contract(size);raw=self.output(size);gain=c['gain'];top=c['pad_tblr'][0]
            raw[0,:,0]=[320*gain,180*gain+top,80*gain,80*gain,.9,.8]
            calls=[]
            def native(values,source,model,display,classes,conf,nms,limit):
                calls.append(values.copy())
                self.assertEqual(values.shape,(c['positions'],5));self.assertEqual(values.itemsize,4)
                self.assertEqual(values.strides,(20,4));self.assertEqual(classes,1)
                self.assertEqual((source,model,display),([360,640],[size,size],[360,640]))
                x,y,w,h,score=values[0]
                return [[[(x-w/2)/gain,(y-h/2)/gain,w/gain,h/gain]],[0],[score]]
            result=decode_two_class_proposals(raw,c,native=native,np=np)
            self.assertEqual(len(calls),2);self.assertEqual(len(result['proposals']),2)
            for p in result['proposals']:
                np.testing.assert_allclose(p['bbox_xywh'],[280,140,80,80],atol=1e-4)
                self.assertTrue(p['proposal_only']);self.assertFalse(p['geometry_verified'])
            self.assertEqual([p['class_id'] for p in result['proposals']],[0,1])
            self.assertIsNone(result['official_points']);self.assertFalse(result['physical_action'])

    def test_four_tops_are_retained_for_geometry_and_saturation_is_not_complete(self):
        calls=[0]
        def native(*args):
            class_id=calls[0];calls[0]+=1
            return [[[20*i,30,10,10] for i in range(4 if class_id==0 else 1)],
                    [0]*(4 if class_id==0 else 1),[.8]*(4 if class_id==0 else 1)]
        result=decode_two_class_proposals(self.output(),input_contract(),limits=(4,16),native=native,np=np)
        self.assertFalse(result['complete']);self.assertEqual(result['per_class_saturated'],[True,False])
        self.assertEqual(sum(p['class_id']==0 for p in result['proposals']),4)
        self.assertEqual(sum(p['class_id']==1 for p in result['proposals']),1)

    def test_invalid_shapes_scores_or_transforms_reject_before_native_pointer_read(self):
        c=input_contract();cases=[]
        raw=self.output();raw[0,4,0]=float('nan');cases.append(raw)
        raw=self.output();raw[0,5,0]=1.1;cases.append(raw)
        raw=self.output();raw[0,2,0]=-1;cases.append(raw)
        cases.extend([np.zeros((1,84,2100),np.float32),np.zeros((1,6,8),np.float32)])
        large=np.zeros((1,6,2100),np.float64);large[0,0,0]=1e100;cases.append(large)
        def native(*args):raise AssertionError('Must not enter C with invalid contract')
        for raw in cases:
            with self.subTest(shape=raw.shape),self.assertRaises(ValueError):
                decode_two_class_proposals(raw,c,native=native,np=np)
        bad=dict(c,source_wh=[320,180])
        with self.assertRaises(ValueError):decode_two_class_proposals(self.output(),bad,native=native,np=np)

    def test_padding_and_four_sided_crossing_are_intersected_before_native_clamping(self):
        contract=input_contract(320);raw=self.output();gain=contract['gain'];pad=contract['pad_tblr'][0]
        # First box lies wholly in padding; the other four cross source sides.
        raw[0,:,0]=[160,30,40,20,.9,0]
        sources=([320,0,80,120],[0,180,80,80],[640,180,80,80],[320,360,80,80])
        for i,(x,y,w,h) in enumerate(sources,1):raw[0,:,i]=[x*gain,y*gain+pad,w*gain,h*gain,.9,0]
        def native(values,source,model,display,classes,conf,nms,limit):
            boxes=[];scores=[]
            for x,y,w,h,score in values:
                if score<conf:continue
                boxes.append([max(int((x-w/2)/gain),0),max(int((y-h/2)/gain),0),int(w/gain),int(h/gain)])
                scores.append(score)
            return [boxes,[0]*len(boxes),scores]
        result=decode_two_class_proposals(raw,contract,native=native,np=np)
        self.assertEqual(len(result['proposals']),4)
        expected=([280,0,80,60],[0,140,40,80],[600,140,40,80],[280,320,80,40])
        for proposal,box in zip(result['proposals'],expected):
            self.assertEqual(proposal['bbox_xywh'],list(box));self.assertTrue(proposal['bbox_touches_source_edge'])

    def test_noncontiguous_raw_input_is_copied_to_safe_native_buffer_and_empty_stays_empty(self):
        c=input_contract();raw=np.zeros((1,6,c['positions']*2),np.float64)[:,:,::2]
        def native(values,*args):
            self.assertEqual(values.itemsize,4);self.assertEqual(values.strides,(20,4));return []
        result=decode_two_class_proposals(raw,c,native=native,np=np)
        self.assertTrue(result['complete']);self.assertEqual(result['proposals'],[])


if __name__=='__main__':unittest.main()
