import unittest
from unittest.mock import patch
from types import SimpleNamespace
from itertools import permutations
from vision.top import circle_surfaces,circle_boundary_support
from k230.visual_inspector import VisualInspector
from test_visual_inspector import top_profile
from test_block_service import frame


class TopGeometryTests(unittest.TestCase):
    def test_native_boundary_gate_rejects_closer_false_circle_and_keeps_reference_selection(self):
        class Circle:
            def __init__(self,x):self.cx=x
            def x(self):return self.cx
            def y(self):return 80
            def r(self):return 10
        class Image:
            def width(self):return 320
            def height(self):return 240
            def find_circles(self,**kwargs):return [Circle(110),Circle(160),Circle(210)]
            def get_pixel(self,x,y):
                if any((x-cx)**2+(y-80)**2<=100 for cx in (110,210)):return (22,0,0)
                if 140<=x<=180 and 60<=y<=110:return (22,0,0)
                return (65,0,0)
        cfg=top_profile();top=cfg['TOP_CONFIG']['BUFFER']
        top.update(roi=[80,55,175,145],min_area=120,thresholds={'5':[19,33,-13,9,-13,9]},
            circle_refine_ids=[5],circle_parameters={'threshold':600,'r_min':7,'r_max':16,'r_step':1},
            circle_boundary={'inner_ratio':0.7,'outer_ratio':1.2,'min_l_contrast':5,'min_supported':5})
        with patch.dict('sys.modules',{'image':SimpleNamespace(rgb_to_lab=lambda *v:v)}):
            for reference,expected in (([159,80],110),([161,80],210)):
                top['reference']=reference
                result=VisualInspector('TOP',5,'BUFFER',cfg).process(frame(Image()))
                self.assertEqual(result['u_px'],expected)

    def test_radial_boundary_rejects_uniform_same_lab_false_circle(self):
        params={'inner_ratio':0.7,'outer_ratio':1.2,'min_l_contrast':5,'min_supported':6}
        class Image:
            def width(self):return 320
            def height(self):return 240
            def get_pixel(self,x,y):
                # Two real circular boundaries plus a uniform dark rectangle.
                for cx in (110,210):
                    if (x-cx)**2+(y-80)**2<=100:return (22,0,0)
                if 140<=x<=180 and 60<=y<=110:return (22,0,0)
                return (65,0,0)
        for x,expected in ((110,True),(210,True),(160,False)):
            result=circle_boundary_support(Image(),lambda *v:v,x,80,10,params)
            self.assertEqual(result['verified'],expected)
        params['min_supported']=True
        with self.assertRaises(ValueError):circle_boundary_support(Image(),lambda *v:v,110,80,10,params)

    def test_all_native_candidate_orders_produce_identical_groups(self):
        rows=[{'x':x,'y':80,'r':10,'interior_verified':True} for x in (100,104,110)]
        expected=circle_surfaces(rows,5)
        for order in permutations(rows):self.assertEqual(circle_surfaces(order,5),expected)

    def test_invalid_profile_or_adapter_parameters_never_reach_native(self):
        from k230.vision_adapter import VisionAdapter
        for bad in ({'r_step':0},{'r_min':True},{'r_max':6},{'threshold':float('nan')},{'x_stride':1.5}):
            cfg=top_profile();top=cfg['TOP_CONFIG']['BUFFER']
            top['circle_refine_ids']=[5];top['thresholds']={'5':[19,33,-13,9,-13,9]}
            top['circle_parameters']=dict(threshold=600,r_min=7,r_max=16,r_step=1,**{})
            top['circle_parameters'].update(bad)
            with self.assertRaises(ValueError):VisualInspector('TOP',5,'BUFFER',cfg)
            top['thresholds']={5:top['thresholds']['5']}
            cfg.update(CAMERA_IDS=(1,2),MODE_CAMERA={'TOP':1})
            native=SimpleNamespace(width=lambda:320,height=lambda:240,
                find_circles=lambda **kwargs:self.fail('Invalid parameters reached native'))
            result=VisionAdapter(cfg).detect('TOP',frame(native),5,'BUFFER')
            self.assertEqual(result['status'],'ERROR');self.assertIsNone(result['u_px'])

    def test_raw_roi_crossing_circle_cannot_shift_valid_neighbor_and_reference_selects_two_objects(self):
        class Circle:
            def __init__(self,x,r):self.cx=x;self.radius=r
            def x(self):return self.cx
            def y(self):return 80
            def r(self):return self.radius
        circles=[Circle(90,12),Circle(96,8)]
        native=SimpleNamespace(width=lambda:320,height=lambda:240,
            find_circles=lambda **kwargs:circles,get_pixel=lambda x,y:(22,0,0))
        cfg=top_profile();top=cfg['TOP_CONFIG']['BUFFER']
        top.update(roi=[80,55,175,145],min_area=120,thresholds={'5':[19,33,-13,9,-13,9]},
            circle_refine_ids=[5],circle_parameters={'threshold':600,'r_min':7,'r_max':16,'r_step':1})
        with patch.dict('sys.modules',{'image':SimpleNamespace(rgb_to_lab=lambda *v:v)}):
            inspector=VisualInspector('TOP',5,'BUFFER',cfg)
            result=inspector.process(frame(native));self.assertEqual((result['u_px'],result['v_px']),(96,80))
            circles[:]=[Circle(110,10),Circle(210,10)]
            for reference,expected in (([220,80],210),([100,80],110)):
                top['reference']=reference
                result=VisualInspector('TOP',5,'BUFFER',cfg).process(frame(native))
                self.assertEqual(result['u_px'],expected)

    def test_wrong_interior_and_invalid_geometry_do_not_create_top_center(self):
        rows=[{'x':10,'y':20,'r':9,'interior_verified':False},
              {'x':10,'y':20,'r':-1,'interior_verified':True},
              {'x':float('nan'),'y':20,'r':9,'interior_verified':True}]
        self.assertEqual(circle_surfaces(rows,5),[])
        with self.assertRaises(ValueError):circle_surfaces([],5,0)

    def test_nearby_circles_cluster_but_two_objects_do_not_chain_merge(self):
        rows=[{'x':x,'y':20,'r':10,'interior_verified':True} for x in (10,14,20,60)]
        result=circle_surfaces(rows,5,6)
        self.assertEqual(len(result),3)
        self.assertEqual(result[0]['circle_support'],2)
        x,y,w,h=result[0]['bbox']
        self.assertEqual((x+w/2,y+h/2),(12,20))
        self.assertTrue(all(r['target_id']==5 for r in result))

    def test_native_adapter_path_uses_current_circle_interior_and_refuses_body_skin_or_loss(self):
        class Circle:
            def __init__(self,x,y,r):self.point=(x,y,r)
            def x(self):return self.point[0]
            def y(self):return self.point[1]
            def r(self):return self.point[2]
        class Image:
            def __init__(self):self.circles=[Circle(140,80,10),Circle(142,82,10),Circle(140,110,10),Circle(180,80,10)]
            def width(self):return 320
            def height(self):return 240
            def find_circles(self,**kwargs):return self.circles
            def find_blobs(self,*a,**k):raise AssertionError('Merged blobs must not determine circle center')
            def get_pixel(self,x,y):
                if y>100:return (8,0,0)
                if x>170:return (40,20,20)
                return (22,0,0)
        cfg=top_profile();top=cfg['TOP_CONFIG']['BUFFER'];top['thresholds']={'5':[19,33,-13,9,-13,9]}
        top['circle_refine_ids']=[5];top['circle_parameters']={'threshold':600,'r_min':7,'r_max':16,'r_step':1}
        inspector=VisualInspector('TOP',5,'BUFFER',cfg);im=Image()
        with patch.dict('sys.modules',{'image':SimpleNamespace(rgb_to_lab=lambda *v:v)}):
            for i in range(1,4):result=inspector.process(frame(im,i))
            self.assertEqual((result['u_px'],result['v_px']),(141,81))
            self.assertTrue(result['stable']);self.assertEqual(result['top_geometry'],'LAB_INTERIOR_CONFIRMED_CIRCLE')
            self.assertIsNone(result['x_mm']);self.assertFalse(result['target_pose_valid'])
            im.circles=[];result=inspector.process(frame(im,4))
            self.assertEqual(result['status'],'NOT_FOUND');self.assertIsNone(result['u_px'])
            im.circles=[Circle(86,80,12)];top['roi']=[80,55,175,145]
            inspector=VisualInspector('TOP',5,'BUFFER',cfg)
            result=inspector.process(frame(im,5));self.assertIsNone(result['u_px'])


if __name__=='__main__':unittest.main()
