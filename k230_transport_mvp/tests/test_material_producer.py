import math
import unittest
from unittest.mock import patch

from k230.tuning.material import (MaterialSnapshotProducer, SourceClock, scan_state)
from k230.tuning.schema import defaults,clone,image_profile
from ports.protocol import StreamParser,encode_material_snapshot
from vision.material_tracker import MaterialTracker


def fit(color=1,u=150,v=150):
    return dict(color_id=color,center=[u,v],semiaxes=[20,16],angle_rad=.2,
                accepted=True,reason='GEOMETRY_ONLY')


def configured():
    value=defaults()
    for c in range(1,7):value['contexts']['TOP_BUFFER']['thresholds'][str(c)]=[0,30,-5,5,-5,5]
    return value


def measured(fits=()):
    return dict(material_fits=list(fits),raw_material_fit_count=len(fits),
                material_complete=True,material_limit_exceeded=False,material_algorithm_revision=0,
                material_scan_colors=[1,2,3,4,5,6],material_scan_reason='MATERIAL_SCAN_COMPLETE',
                material_by_color=[dict(color_id=i,scan_complete=True,scan_reason='COMPLETE',threshold_available=True,
                                       top_count=sum(row.get('color_id')==i for row in fits))
                                   for i in range(1,7)])


def frame(number=1,time=100,active=None,**changes):
    active=active or configured()
    result=dict(camera_id=2,frame_id=number,timestamp_ms=time,is_simulated=False,
                image_profile=image_profile(active),active_revision=active['revision'],
                width=active['binding']['width'],height=active['binding']['height'])
    result.update(changes)
    return result


def producer(active=None,draft=None,colors=(1,2,3,4,5,6),**kwargs):
    return MaterialSnapshotProducer(active or configured(),draft or configured(),'b'*32,
                                    list(colors),'SCALAR_FULL_BOUNDARY',**kwargs)


def object_row(color=1,u=150,a=20,b=16):
    return dict(color_id=color,u_px=u,v_px=150,semimajor_px=a,semiminor_px=b,
                angle_rad=.2,score_hint=None,geometry_valid=True,stable=False)


class MaterialProducerTests(unittest.TestCase):
    def test_complete_three_same_color_new_frames_become_stable_and_roundtrip(self):
        p=producer()
        geometry=measured([fit(1,100),fit(1,220),fit(5,340)])
        for n in range(1,4):
            value=p.build(frame(n,n*100),geometry)
            self.assertEqual([o['stable'] for o in value['objects']],[n==3]*3)
            self.assertEqual(value['status'],'OK')
            decoded=StreamParser().feed(encode_material_snapshot(n,value))[0]
            self.assertEqual(decoded['objects'],value['objects'])
        self.assertNotIn('stable',geometry['material_fits'][0])

    def test_complete_zero_and_incomplete_never_mix(self):
        p=producer()
        good=p.build(frame(),measured())
        self.assertEqual(good['status'],'NOT_FOUND')
        cases=[{},dict(measured(),material_complete=False),
               dict(measured(),raw_material_fit_count=1),
               dict(measured(),material_limit_exceeded=True,raw_material_fit_count=4),
               measured([fit(1,100),fit(2,200),fit(3,300),fit(4,400)])]
        for n,geometry in enumerate(cases,2):
            value=p.build(frame(n,n*100),geometry)
            self.assertEqual(value['status'],'UNKNOWN')
            self.assertFalse(value['complete'])
            self.assertEqual(value['objects'],[])
            self.assertTrue(StreamParser().feed(encode_material_snapshot(n,value)))

    def test_missing_color_threshold_and_edge_capacity_do_not_become_empty_scene(self):
        for reason in ('MATERIAL_THRESHOLD_REQUIRED','COMPONENT_LIMIT','EDGE_PROPOSAL_LIMIT'):
            geometry=measured()
            geometry['material_by_color'][4].update(scan_complete=False,scan_reason=reason)
            value=producer().build(frame(),geometry)
            self.assertEqual(value['status'],'UNKNOWN')
            self.assertEqual(value['reason'],reason)
        for colors in ((5,),(),(1,2,3,4,5)):
            self.assertEqual(producer(colors=colors).build(frame(),measured())['status'],'UNKNOWN')

    def test_pixel_binding_source_and_algorithm_must_match(self):
        p=producer()
        for n,change in enumerate((dict(camera_id=1),dict(width=320),dict(active_revision=1),
                                   dict(image_profile=dict(image_profile(defaults()),hmirror=True))),1):
            value=p.build(frame(n,n*100,**change),measured([fit()]))
            self.assertEqual(value['status'],'UNKNOWN')
            self.assertEqual(value['objects'],[])
        geometry=measured([fit()]);geometry['material_algorithm_revision']=99
        self.assertEqual(p.build(frame(5,500),geometry)['reason'],'ALGORITHM_BINDING_MISMATCH')
        draft=configured();draft['binding']['hmirror']=True
        self.assertEqual(producer(draft=draft).build(frame(),measured())['reason'],
                         'DRAFT_CAPTURE_GEOMETRY_MISMATCH')

    def test_frozen_contract_hashes_both_profiles_and_canonical_order(self):
        active=configured();draft=configured();p=producer(active,draft)
        original=p.profile_id
        reordered={key:active[key] for key in reversed(list(active))}
        self.assertEqual(producer(reordered,draft).profile_id,original)
        active['camera']['exposure_us']+=1
        self.assertNotEqual(producer(active,draft).profile_id,original)
        draft['contexts']['TOP_BUFFER']['thresholds']['5']=[0,20,-5,5,-5,5]
        self.assertNotEqual(producer(configured(),draft).profile_id,original)
        self.assertEqual(p.profile_id,original)
        self.assertEqual(p.draft['contexts']['TOP_BUFFER']['thresholds']['5'],[0,30,-5,5,-5,5])
        copy=p.contract();copy['image_profile']['width']=7
        self.assertEqual(p.contract()['image_profile']['width'],640)

    def test_invalid_measured_geometry_and_cross_color_duplicates_are_unknown(self):
        bad=[dict(fit(),accepted=False),dict(fit(),reason='MODEL_PROPOSAL'),
             dict(fit(),center=[float('nan'),150]),dict(fit(),center=[2,150]),
             dict(fit(),semiaxes=[16,20]),dict(fit(),angle_rad=math.pi),dict(fit(),color_id=True)]
        for row in bad:
            self.assertEqual(producer().build(frame(),measured([row]))['status'],'UNKNOWN')
        value=producer().build(frame(),measured([fit(1,150),fit(2,151)]))
        self.assertEqual(value['reason'],'AMBIGUOUS_DUPLICATE_TOP')
        self.assertEqual(value['objects'],[])

    def test_claimed_complete_cannot_override_missing_threshold_or_color_counts(self):
        missing=configured();missing['contexts']['TOP_BUFFER']['thresholds']['5']=None
        value=producer(draft=missing).build(frame(),measured())
        self.assertEqual(value['reason'],'MATERIAL_THRESHOLD_REQUIRED')
        for changes in (dict(top_count=1),dict(threshold_available=False),dict(top_count=True)):
            g=measured();g['material_by_color'][0].update(changes)
            self.assertEqual(producer().build(frame(),g)['status'],'UNKNOWN')

    def test_unknown_sim_and_missing_object_break_temporal_evidence(self):
        p=producer()
        for n in range(1,4):value=p.build(frame(n,n*100),measured([fit()]))
        self.assertTrue(value['objects'][0]['stable'])
        self.assertEqual(p.build(frame(4,400),{})['status'],'UNKNOWN')
        self.assertFalse(p.build(frame(5,500),measured([fit()]))['objects'][0]['stable'])
        value=p.build(frame(6,600,is_simulated=True),measured([fit()]))
        self.assertTrue(value['is_simulated']);self.assertFalse(value['objects'][0]['stable'])
        self.assertFalse(p.build(frame(7,700),measured([fit()]))['objects'][0]['stable'])
        self.assertEqual(p.build(frame(8,800),measured())['objects'],[])
        self.assertFalse(p.build(frame(9,900),measured([fit()]))['objects'][0]['stable'])

    def test_replay_backward_source_never_lowers_high_water_or_reuses_stability(self):
        p=producer()
        for n in range(10,13):p.build(frame(n,n*100),measured([fit()]))
        for packet in (frame(12,1200),frame(1,1300),frame(13,1100),frame(13,1200)):
            with self.assertRaises(ValueError):p.build(packet,measured([fit()]))
        self.assertEqual(p.last_frame,12)
        self.assertFalse(p.build(frame(13,1300),measured([fit()]))['objects'][0]['stable'])
        for invalid in (frame(number=True),frame(time=None),frame(is_simulated=None)):
            with self.assertRaises(ValueError):producer().build(invalid,measured())


class MaterialTrackerTests(unittest.TestCase):
    def test_reordering_same_color_is_by_unique_position_not_list_slot(self):
        t=MaterialTracker()
        first=[object_row(1,100),object_row(1,250),object_row(5,400)]
        t.update('a',1,100,first)
        t.update('a',2,200,list(reversed(first)))
        value=t.update('a',3,300,first[1:]+first[:1])
        self.assertTrue(all(row['stable'] for row in value))
        self.assertEqual([o['u_px'] for o in value],[250,400,100])

    def test_ambiguous_matching_split_merge_and_color_change_restart(self):
        t=MaterialTracker()
        for n in range(1,4):t.update('a',n,n*100,[object_row(1,100),object_row(1,200)])
        value=t.update('a',4,400,[object_row(1,120),object_row(1,125)])
        self.assertFalse(any(row['stable'] for row in value))
        self.assertFalse(t.update('a',5,500,[object_row(1,122)])[0]['stable'])
        self.assertFalse(t.update('a',6,600,[object_row(2,122)])[0]['stable'])

    def test_motion_size_gap_empty_key_change_and_replay_are_bounded(self):
        t=MaterialTracker()
        for n,u in enumerate((100,103,106),1):
            value=t.update('a',n,n*100,[object_row(u=u)])
        self.assertFalse(value[0]['stable'])  # 3px steps do not satisfy a 4px total window.
        for n in range(4,7):value=t.update('a',n,n*100,[object_row(u=106)])
        self.assertTrue(value[0]['stable'])
        self.assertFalse(t.update('a',7,700,[object_row(u=106,a=26,b=21)])[0]['stable'])
        self.assertFalse(t.update('a',8,1100,[object_row(u=106)])[0]['stable'])
        t.update('a',9,1200,[])
        self.assertFalse(t.tracks)
        self.assertFalse(t.update('b',10,1300,[object_row()])[0]['stable'])
        with self.assertRaises(ValueError):t.update('b',9,1400,[object_row()])
        self.assertEqual(t.last_frame,10)
        self.assertFalse(t.update('b',11,1500,[object_row()])[0]['stable'])
        self.assertLessEqual(len(t.tracks),3)

    def test_clock_wrap_and_history_bound(self):
        t=MaterialTracker()
        for number,stamp in enumerate((4294967196,0,100),1):
            value=t.update('a',number,stamp,[object_row()])
        self.assertTrue(value[0]['stable'])
        for n in range(4,40):t.update('a',n,(n-2)*100,[object_row()])
        self.assertEqual(len(t.tracks[0]['history']),3)


class MaterialClockAndScanTests(unittest.TestCase):
    def test_declared_native_tick_wrap_extends_once_and_rejects_rollback(self):
        clock=SourceClock(1024)
        self.assertEqual([clock.update(v) for v in (1000,1020,4,20)],[1000,1020,1028,1044])
        for raw in (20,19,800,True,-1,1024):
            with self.assertRaises(ValueError):clock.update(raw)
        self.assertEqual(clock.update(30),1054)
        c=SourceClock()
        self.assertEqual(c.update(4294967290),4294967290)
        self.assertEqual(c.update(2),2)
        with self.assertRaises(ValueError):SourceClock(1000)

    def test_scan_states_need_explicit_per_color_completion(self):
        rows=[dict(color_id=i,scan_complete=True) for i in range(1,7)]
        self.assertTrue(scan_state([1,2,3,4,5,6],rows,True)[0])
        for args in (([],[],True),([1],[],True),([1],[dict(color_id=1)],True),
                     ([1],[dict(color_id=1,scan_complete=True)],False),
                     ([1],[dict(color_id=1,scan_complete=True)],True,True)):
            self.assertFalse(scan_state(*args)[0])


class NativeMaterialScanTests(unittest.TestCase):
    def native(self,contour=None,edge=None,missing=(),blobs=True,selected=True):
        import numpy as np
        from k230.tuning import live
        from vision import ring_native,ellipse_edges
        class Blob:
            def rect(self):return (100,100,40,40)
            def code(self):return 31  # First five, not sixth, ensuring independent empty scan.
        class Image:
            def __init__(self,w=640,h=360):self.array=np.zeros((h,w),np.uint8)
            def copy(self,roi=None):return Image(roi[2],roi[3]) if roi else Image()
            def clear(self):pass
            def draw_image(self,*a,**k):pass
            def to_grayscale(self):return self
            def to_rgb565(self):return self
            def to_numpy_ref(self):return self.array
            def binary(self,*a):return self
            def find_blobs(self,*a,**k):return [Blob()] if blobs else []
            def draw_line(self,*a,**k):pass
            def draw_cross(self,*a,**k):pass
        profile=defaults()
        for i in range(1,7):
            if i not in missing:profile['contexts']['TOP_BUFFER']['thresholds'][str(i)]=[0,30,-5,5,-5,5]
        paper=dict(groups=[],reason='NO_PAPER',edges=np.zeros((360,640),np.uint8))
        contour=contour or dict(complete=True,accepted=[],components=[],reason='EMPTY_CONTOURS')
        edge=edge or dict(complete=True,accepted=[],components=[],reason='EMPTY_EDGES')
        with patch.object(ring_native,'detect_paper_targets',return_value=paper), \
             patch.object(live,'extract_mask_ellipses',side_effect=lambda *a,**k:clone(contour)), \
             patch.object(ellipse_edges,'extract_dark_edge_ellipses',return_value=edge):
            return live.native_views(Image(),profile,'RING','PROCESS',1,selected=selected,
                                     diagnostic_material_colors=[1,2,3,4,5,6],annotated_preview=True)[2]

    def test_full_empty_scan_complete_is_independent_from_paper(self):
        g=self.native(blobs=False)
        self.assertTrue(g['material_complete'])
        self.assertEqual(len(g['material_by_color']),6)
        self.assertTrue(all(row['scan_complete'] for row in g['material_by_color']))
        self.assertEqual(producer().build(frame(),g)['status'],'NOT_FOUND')

    def test_native_incomplete_black_edge_missing_threshold_and_idle_propagate(self):
        cases=[dict(missing=(3,)),dict(selected=False),
               dict(contour=dict(complete=False,accepted=[],components=[],reason='COMPONENT_LIMIT')),
               dict(edge=dict(complete=False,accepted=[],components=[],reason='EDGE_PROPOSAL_LIMIT'))]
        for case in cases:
            g=self.native(**case)
            self.assertFalse(g['material_complete'])
            self.assertEqual(producer().build(frame(),g)['status'],'UNKNOWN')
        black=self.native(edge=dict(complete=False,accepted=[],components=[],reason='EDGE_PROPOSAL_LIMIT'))
        self.assertEqual(black['material_by_color'][4]['scan_reason'],'EDGE_PROPOSAL_LIMIT')


if __name__=='__main__':
    unittest.main()
