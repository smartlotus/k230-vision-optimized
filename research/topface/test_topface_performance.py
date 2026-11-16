"""Exact decode parity for native vector class gating."""
import unittest
import numpy as np
from board.topface_decode import decode,decode_tensor


class VectorDecodeTests(unittest.TestCase):
    def test_finite_raw_parity_across_empty_and_six_colors(self):
        rng=np.random.default_rng(20261116)
        for n in (0,1,3,4,5,6):
            raw=rng.normal(0,.3,(1,1275,50)).astype(np.float32)
            raw[:,:,:6]=-20
            for i in range(n):
                raw[0,100+i*9,i]=5
                raw[0,100+i*9,44:46]=[.6,.8]
            a=decode(lambda p,c:raw[0,p,c])
            self.assertEqual(a,decode_tensor(raw))

    def test_nonfinite_background_class_and_selected_geometry_are_rejected(self):
        for channel in range(6):
            for bad in (float('nan'),float('inf'),-float('inf')):
                raw=np.zeros((1,1275,50),np.float32);raw[:,:,:6]=-20
                raw[0,222,channel]=bad
                with self.assertRaises(ValueError):decode_tensor(raw)
        raw=np.zeros((1,1275,50),np.float32);raw[0,100,42]=float('nan')
        with self.assertRaises(ValueError):decode_tensor(raw)

    def test_equal_scores_and_threshold_boundary_keep_reference_order(self):
        raw=np.zeros((1,1275,50),np.float32);raw[:,:,:6]=-20
        raw[0,100,0:2]=0
        raw[0,109,1]=0
        self.assertEqual(decode(lambda p,c:raw[0,p,c],score_threshold=.5),
                         decode_tensor(raw,score_threshold=.5))


if __name__=='__main__':unittest.main()
